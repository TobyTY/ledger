"""Ledger operations.

``post_transaction`` is the only way value moves, and it is idempotent: one key
posts exactly once, no matter how many times the request arrives or how many
callers race for it. That property is what makes a retry safe -- a client that
times out and retries gets the original transaction back rather than a second
charge.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Entry, Transaction

# Postgres SQLSTATEs we care about. Telling these apart is the whole correctness
# argument below -- see post_transaction.
UNIQUE_VIOLATION = "23505"
CHECK_VIOLATION = "23514"


class Unbalanced(ValueError):
    """Legs do not sum to zero, so this is not a transaction."""


@dataclass(frozen=True)
class Leg:
    """One side of a movement. Debits positive, credits negative, in paise."""

    account_id: int
    amount_minor: int


def _sqlstate(exc: IntegrityError) -> str | None:
    orig = getattr(exc, "orig", None)
    return getattr(orig, "sqlstate", None)


def _assert_balanced(legs: list[Leg]) -> None:
    """Application-level guard, for good error messages.

    This is a courtesy, not the guarantee. The database enforces the same rule
    through a deferred constraint trigger, and the test suite disables this
    function to prove the database still refuses an unbalanced write on its own.
    """
    if len(legs) < 2:
        raise Unbalanced("a transaction needs at least two legs")
    total = sum(leg.amount_minor for leg in legs)
    if total != 0:
        raise Unbalanced(f"legs sum to {total} paise, must sum to 0")


def _find_by_key(session: Session, idempotency_key: str) -> Transaction | None:
    return session.scalar(
        select(Transaction).where(Transaction.idempotency_key == idempotency_key)
    )


def post_transaction(
    session: Session,
    *,
    idempotency_key: str,
    kind: str,
    legs: list[Leg],
) -> tuple[Transaction, bool]:
    """Post a balanced transaction exactly once.

    Returns ``(transaction, created)``. ``created`` is False when this key had
    already been posted, which lets the API answer 200 instead of 201 without
    the caller having to care which of its retries won.

    On the race: two callers can present the same key simultaneously. Postgres
    serialises them on the unique index -- the loser blocks until the winner
    commits, then raises 23505. We roll back and read the winner's row.

    The important subtlety is that we only swallow 23505. A 23514 means the
    deferred balance trigger rejected the transaction at COMMIT, which is a real
    failure and must propagate. Catching IntegrityError broadly here would
    silently convert an unbalanced write into "already posted" and hand the
    caller someone else's transaction -- the exact class of bug that puts a
    ledger out by a few paise and nobody notices for a month.
    """
    _assert_balanced(legs)

    existing = _find_by_key(session, idempotency_key)
    if existing is not None:
        return existing, False

    txn = Transaction(idempotency_key=idempotency_key, kind=kind)
    session.add(txn)
    try:
        # A duplicate key surfaces here: the unique index is checked immediately.
        session.flush()
        for leg in legs:
            session.add(
                Entry(
                    transaction_id=txn.id,
                    account_id=leg.account_id,
                    amount_minor=leg.amount_minor,
                )
            )
        # The balance trigger is DEFERRABLE INITIALLY DEFERRED, so an unbalanced
        # transaction surfaces here at COMMIT, not at the flush above.
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        if _sqlstate(exc) == UNIQUE_VIOLATION:
            winner = _find_by_key(session, idempotency_key)
            if winner is not None:
                return winner, False
        raise

    return txn, True


def account_balance(session: Session, account_id: int) -> int:
    """Balance in paise, derived by summing entries.

    Deliberately not a stored column. A cached balance is a second source of
    truth that can drift from the entries; deriving it means the entries are the
    only truth there is. The reconciliation job exists to prove that stays true
    once a stored balance is introduced for performance.
    """
    return session.scalar(
        select(func.coalesce(func.sum(Entry.amount_minor), 0)).where(
            Entry.account_id == account_id
        )
    )
