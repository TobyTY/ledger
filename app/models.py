"""Ledger schema.

Two rules the rest of the service depends on:

1. Money is a signed integer count of minor units (paise). Never a float, never
   a Decimal column. A float ledger is wrong by construction -- 0.1 + 0.2 is not
   0.3 in binary floating point, and a ledger that cannot represent its own
   balances exactly is not a ledger.

2. Entries carry a *signed* amount. Debits are positive, credits negative, and
   the entries belonging to one transaction must sum to exactly zero. That
   invariant is enforced in the database (see the deferred constraint trigger in
   the initial migration), not here, so no application bug can violate it.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    CHAR,
    BigInteger,
    Enum,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TIMESTAMP


class Base(DeclarativeBase):
    pass


class AccountType(str, enum.Enum):
    """Account taxonomy.

    User accounts hold the customer's assets; house accounts are ours. Every
    transaction moves value between them, and the two sides must net to zero --
    which is what makes the reconciliation job possible.
    """

    user_cash = "user_cash"
    user_securities = "user_securities"
    house_fees = "house_fees"
    house_settlement = "house_settlement"


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[AccountType] = mapped_column(
        Enum(AccountType, name="account_type", native_enum=True), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str] = mapped_column(CHAR(3), nullable=False, server_default="INR")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class Transaction(Base):
    """One atomic movement of value, made of two or more entries.

    ``idempotency_key`` is unique. A caller that retries after a timeout sends
    the same key and gets the original transaction back rather than posting a
    second one -- the difference between a dropped response and a double charge.
    """

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class Entry(Base):
    """A single leg of a transaction. Append-only.

    There is deliberately no update or delete path. A mistake is corrected by
    posting a compensating entry, so the full history stays reconstructible --
    you can replay any account's balance as of any point in time.
    """

    __tablename__ = "entries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.id"), nullable=False
    )
    account_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("accounts.id"), nullable=False
    )
    amount_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_entries_account_id", "account_id"),
        Index("ix_entries_transaction_id", "transaction_id"),
    )
