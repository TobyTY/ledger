"""The four invariants the ledger rests on.

If any of these fail the service is not a ledger, it is a table of numbers.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import Entry, Transaction


def test_balanced_transaction_commits(session, cash_account, settlement_account, idem_key):
    """Two legs that sum to zero are accepted."""
    txn = Transaction(idempotency_key=idem_key, kind="deposit")
    session.add(txn)
    session.flush()

    # ₹1,000.00 in, booked as 100000 paise.
    session.add(Entry(transaction_id=txn.id, account_id=cash_account.id, amount_minor=100_000))
    session.add(
        Entry(transaction_id=txn.id, account_id=settlement_account.id, amount_minor=-100_000)
    )
    session.commit()

    total = session.execute(
        text("SELECT COALESCE(SUM(amount_minor), 0) FROM entries WHERE transaction_id = :t"),
        {"t": txn.id},
    ).scalar_one()
    assert total == 0


def test_unbalanced_transaction_fails_at_commit_not_at_insert(
    session, cash_account, idem_key
):
    """The timing is the point, not just the failure.

    A plain AFTER INSERT trigger would reject the first leg of a valid two-leg
    transaction, because at that instant the transaction genuinely does not
    balance. DEFERRABLE INITIALLY DEFERRED moves the check to COMMIT, so the
    insert must succeed and the commit must fail. This test asserts both halves
    of that, which is what proves the constraint is deferred rather than merely
    present.
    """
    txn = Transaction(idempotency_key=idem_key, kind="deposit")
    session.add(txn)
    session.flush()

    session.add(Entry(transaction_id=txn.id, account_id=cash_account.id, amount_minor=100_000))

    # Must NOT raise: the constraint is deferred, so a half-written transaction
    # is legal right up until commit.
    session.flush()

    with pytest.raises(IntegrityError) as exc_info:
        session.commit()

    assert "does not balance" in str(exc_info.value)


def test_entries_are_append_only(session, cash_account, settlement_account, idem_key):
    """History cannot be rewritten -- corrections are compensating entries."""
    txn = Transaction(idempotency_key=idem_key, kind="deposit")
    session.add(txn)
    session.flush()
    session.add(Entry(transaction_id=txn.id, account_id=cash_account.id, amount_minor=100_000))
    session.add(
        Entry(transaction_id=txn.id, account_id=settlement_account.id, amount_minor=-100_000)
    )
    session.commit()

    with pytest.raises(DBAPIError) as exc_info:
        session.execute(text("UPDATE entries SET amount_minor = 1 WHERE id = 1"))
        session.commit()
    assert "append-only" in str(exc_info.value)
    session.rollback()

    with pytest.raises(DBAPIError) as exc_info:
        session.execute(text("DELETE FROM entries WHERE id = 1"))
        session.commit()
    assert "append-only" in str(exc_info.value)
    session.rollback()


def test_money_is_integer_minor_units(session, cash_account, settlement_account, idem_key):
    """Amounts are exact, because they are integers.

    The first assertion is the reason the column is BIGINT: binary floating
    point cannot represent these values, and a ledger that cannot represent its
    own balances exactly is not a ledger.
    """
    assert 0.1 + 0.2 != 0.3

    column_type = session.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'entries' AND column_name = 'amount_minor'"
        )
    ).scalar_one()
    assert column_type == "bigint"

    # ₹1,234.56 -> 123456 paise. Round-trips exactly, every time.
    txn = Transaction(idempotency_key=idem_key, kind="deposit")
    session.add(txn)
    session.flush()
    session.add(Entry(transaction_id=txn.id, account_id=cash_account.id, amount_minor=123_456))
    session.add(
        Entry(transaction_id=txn.id, account_id=settlement_account.id, amount_minor=-123_456)
    )
    session.commit()

    balance = session.execute(
        text("SELECT SUM(amount_minor) FROM entries WHERE account_id = :a"),
        {"a": cash_account.id},
    ).scalar_one()
    assert balance == 123_456
