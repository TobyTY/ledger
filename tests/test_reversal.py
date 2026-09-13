"""Reversals: undoing a transaction without editing history.

The interesting tests are the refusals. A reversal that works is arithmetic; a
reversal that happens twice, or that unwinds one half of a hold, is money.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from app.db import SessionLocal
from app.holds import capture_hold, place_hold
from app.ledger import (
    Leg,
    ReversalRefused,
    account_balance,
    post_transaction,
    reverse_transaction,
)
from app.models import Entry, Transaction


@pytest.fixture
def deposit(session, cash_account, settlement_account):
    txn, _ = post_transaction(
        session,
        idempotency_key=f"dep-{uuid.uuid4()}",
        kind="deposit",
        legs=[Leg(cash_account.id, 5_000_00), Leg(settlement_account.id, -5_000_00)],
    )
    return txn


def test_a_reversal_returns_every_account_to_where_it_started(
    session, deposit, cash_account, settlement_account
):
    assert account_balance(session, cash_account.id) == 5_000_00

    reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-1"
    )

    assert account_balance(session, cash_account.id) == 0
    assert account_balance(session, settlement_account.id) == 0


def test_the_original_stays_in_history(session, deposit):
    """Append-only means a correction adds a row; it never removes one.

    This is what keeps any account's balance reconstructible as of any past
    moment.
    """
    reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-1"
    )

    original_legs = session.scalars(
        __import__("sqlalchemy").select(Entry).where(Entry.transaction_id == deposit.id)
    ).all()
    assert len(original_legs) == 2
    assert session.get(Transaction, deposit.id) is not None


def test_the_reversal_points_at_what_it_reversed(session, deposit):
    reversal, created = reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-1"
    )
    assert created
    assert reversal.reverses_transaction_id == deposit.id
    assert "reversal" in reversal.kind


def test_reversing_twice_reverses_once(session, deposit, cash_account):
    """Without this the account is over-credited by exactly the original
    amount — and every transaction involved still balances perfectly, so no
    per-transaction audit would ever see it."""
    first, created_first = reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-1"
    )
    second, created_second = reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-2"
    )

    assert created_first and not created_second
    assert first.id == second.id
    assert account_balance(session, cash_account.id) == 0


def test_concurrent_reversals_reverse_once(session, deposit, cash_account):
    """The check-then-insert both callers pass. Only the unique index stops it."""
    transaction_id = deposit.id
    barrier = threading.Barrier(6)
    created_flags: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        barrier.wait()
        try:
            with SessionLocal() as s:
                _, created = reverse_transaction(
                    s, transaction_id=transaction_id, idempotency_key=f"rev-{n}"
                )
            with lock:
                created_flags.append(created)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"
    assert sum(created_flags) == 1, f"{sum(created_flags)} reversals were created"

    session.expire_all()
    assert account_balance(session, cash_account.id) == 0


def test_a_reversal_can_itself_be_reversed(session, deposit, cash_account):
    """Legitimate: somebody reversed the wrong transaction."""
    reversal, _ = reverse_transaction(
        session, transaction_id=deposit.id, idempotency_key="rev-1"
    )
    reverse_transaction(
        session, transaction_id=reversal.id, idempotency_key="rev-of-rev"
    )
    assert account_balance(session, cash_account.id) == 5_000_00


def test_a_hold_placement_cannot_be_reversed_generically(
    session, cash_account, settlement_account
):
    """The refusal that matters.

    Reversing the placement would put the funds back in spendable cash while
    the hold row still reads 'active'. The money would be simultaneously
    spendable and claimed, and `held_vs_holds` would report a discrepancy it
    could not explain. Holds are undone through release, which moves the state
    machine with the money.
    """
    post_transaction(
        session,
        idempotency_key="fund",
        kind="deposit",
        legs=[Leg(cash_account.id, 5_000_00), Leg(settlement_account.id, -5_000_00)],
    )
    result = place_hold(
        session, idempotency_key="h1", cash_account=cash_account, amount_minor=1_000_00
    )

    with pytest.raises(ReversalRefused):
        reverse_transaction(
            session,
            transaction_id=result.hold.place_transaction_id,
            idempotency_key="rev-hold",
        )


def test_a_hold_capture_cannot_be_reversed_generically(
    session, cash_account, settlement_account
):
    post_transaction(
        session,
        idempotency_key="fund",
        kind="deposit",
        legs=[Leg(cash_account.id, 5_000_00), Leg(settlement_account.id, -5_000_00)],
    )
    result = place_hold(
        session, idempotency_key="h1", cash_account=cash_account, amount_minor=1_000_00
    )
    hold, _ = capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key="c1",
    )

    with pytest.raises(ReversalRefused):
        reverse_transaction(
            session,
            transaction_id=hold.resolve_transaction_id,
            idempotency_key="rev-capture",
        )


def test_reversing_a_transaction_that_does_not_exist_is_an_error(session):
    with pytest.raises(ValueError):
        reverse_transaction(
            session, transaction_id=9_999_999, idempotency_key="rev-nothing"
        )
