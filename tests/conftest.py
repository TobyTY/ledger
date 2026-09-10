from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.db import SessionLocal, engine
from app.models import Account, AccountType


@pytest.fixture
def session():
    """A session on a clean-ish database.

    Entries are append-only by design, so tests cannot DELETE through the ORM --
    the trigger would reject it. Truncation goes through a raw statement with
    the mutation trigger disabled, which is the one place bypassing it is
    legitimate.
    """
    with SessionLocal() as s:
        s.execute(text("ALTER TABLE entries DISABLE TRIGGER entries_no_mutation"))
        s.execute(text("TRUNCATE entries, transactions, accounts RESTART IDENTITY CASCADE"))
        s.execute(text("ALTER TABLE entries ENABLE TRIGGER entries_no_mutation"))
        s.commit()
        yield s
        s.rollback()


@pytest.fixture
def cash_account(session) -> Account:
    account = Account(type=AccountType.user_cash, owner_id="user-1", currency="INR")
    session.add(account)
    session.commit()
    return account


@pytest.fixture
def settlement_account(session) -> Account:
    account = Account(type=AccountType.house_settlement, currency="INR")
    session.add(account)
    session.commit()
    return account


@pytest.fixture
def idem_key() -> str:
    return f"test-{uuid.uuid4()}"
