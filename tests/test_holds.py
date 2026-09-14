"""The hold lifecycle, including the ways it is asked to go wrong.

Everything here runs against real Postgres, because the properties being tested
are properties of Postgres: row-level locking under concurrent UPDATE, a CHECK
constraint refusing an inconsistent resolution, a deferred trigger firing at
COMMIT. A fake would agree with whatever this code did.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import DatabaseError, IntegrityError, InternalError

from app.db import SessionLocal
from app.holds import (
    HoldError,
    HoldNotClaimable,
    InsufficientFunds,
    available_balance,
    capture_hold,
    expire_due_holds,
    held_account_for,
    held_balance,
    place_hold,
    release_hold,
)
from app.ledger import Leg, post_transaction
from app.models import Hold, HoldState


def fund(session, cash_account, settlement_account, paise: int) -> None:
    """Put money in a customer's account, from the house."""
    post_transaction(
        session,
        idempotency_key=f"fund-{uuid.uuid4()}",
        kind="deposit",
        legs=[Leg(cash_account.id, paise), Leg(settlement_account.id, -paise)],
    )


@pytest.fixture
def funded(session, cash_account, settlement_account):
    fund(session, cash_account, settlement_account, 10_000_00)
    return cash_account


# ---------------------------------------------------------------------------
# Placing
# ---------------------------------------------------------------------------


def test_a_hold_moves_money_out_of_spendable_cash(
    session, funded, settlement_account, idem_key
):
    """The property the whole design rests on.

    Available balance needs no knowledge of holds, because held money is not
    in the cash account any more.
    """
    before = available_balance(session, funded.id)

    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )

    assert result.created
    assert available_balance(session, funded.id) == before - 3_000_00
    assert held_balance(session, result.hold.held_account_id) == 3_000_00


def test_the_customer_still_owns_the_money(session, funded, idem_key):
    """Held cash left the spendable account; it did not leave the customer."""
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    total = available_balance(session, funded.id) + held_balance(
        session, result.hold.held_account_id
    )
    assert total == 10_000_00


def test_a_hold_beyond_the_balance_is_refused(session, funded, idem_key):
    with pytest.raises(InsufficientFunds):
        place_hold(
            session,
            idempotency_key=idem_key,
            cash_account=funded,
            amount_minor=10_000_01,
        )


def test_two_holds_cannot_both_use_the_same_money(session, funded):
    """Sequentially this is the check working. The concurrent version is
    `no_overdrawn_cash` in the reconciliation job, which exists precisely
    because this check cannot be made atomic without serialising every write."""
    place_hold(
        session, idempotency_key="h1", cash_account=funded, amount_minor=6_000_00
    )
    with pytest.raises(InsufficientFunds):
        place_hold(
            session, idempotency_key="h2", cash_account=funded, amount_minor=6_000_00
        )


def test_placing_the_same_hold_twice_places_it_once(session, funded, idem_key):
    first = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=1_000_00
    )
    second = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=1_000_00
    )

    assert first.created and not second.created
    assert first.hold.id == second.hold.id
    assert available_balance(session, funded.id) == 9_000_00


def test_a_zero_or_negative_hold_is_refused(session, funded, idem_key):
    with pytest.raises(HoldError):
        place_hold(
            session, idempotency_key=idem_key, cash_account=funded, amount_minor=0
        )


# ---------------------------------------------------------------------------
# Capturing
# ---------------------------------------------------------------------------


def test_a_full_capture_settles_the_whole_hold(
    session, funded, settlement_account, idem_key
):
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    hold, changed = capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
    )

    assert changed
    assert hold.state == HoldState.captured
    assert hold.captured_minor == 3_000_00
    assert held_balance(session, hold.held_account_id) == 0
    assert available_balance(session, funded.id) == 7_000_00


def test_a_partial_capture_returns_the_remainder_immediately(
    session, funded, settlement_account, idem_key
):
    """An order that fills for less than it reserved.

    The difference must not be left in the holding account: it would be money
    the customer owns, cannot spend, and has no open order against, and nothing
    else in the system would ever notice.
    """
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    hold, _ = capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
        amount_minor=1_200_00,
    )

    assert hold.captured_minor == 1_200_00
    assert held_balance(session, hold.held_account_id) == 0
    # 10,000 - 3,000 held + 1,800 returned
    assert available_balance(session, funded.id) == 8_800_00


def test_a_zero_capture_is_a_release(
    session, funded, settlement_account, idem_key
):
    """An order that was accepted and filled for nothing."""
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
        amount_minor=0,
    )
    assert available_balance(session, funded.id) == 10_000_00


def test_capturing_more_than_the_hold_is_refused(
    session, funded, settlement_account, idem_key
):
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    with pytest.raises(HoldError):
        capture_hold(
            session,
            hold=result.hold,
            settlement_account=settlement_account,
            idempotency_key=idem_key,
            amount_minor=3_000_01,
        )
    assert result.hold.state == HoldState.active


def test_the_same_fill_arriving_twice_moves_money_once(
    session, funded, settlement_account, idem_key
):
    """Exchange fill callbacks are at-least-once. A repeat is the normal case."""
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    _, first = capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
        amount_minor=1_200_00,
    )
    _, second = capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
        amount_minor=1_200_00,
    )

    assert first and not second
    assert available_balance(session, funded.id) == 8_800_00


def test_a_second_fill_for_a_different_amount_is_refused(
    session, funded, settlement_account, idem_key
):
    """Not the same callback arriving again — a contradiction, and money has
    already moved on the first one."""
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    capture_hold(
        session,
        hold=result.hold,
        settlement_account=settlement_account,
        idempotency_key=idem_key,
        amount_minor=1_200_00,
    )
    with pytest.raises(HoldNotClaimable):
        capture_hold(
            session,
            hold=result.hold,
            settlement_account=settlement_account,
            idempotency_key=f"{idem_key}-b",
            amount_minor=2_000_00,
        )


def test_a_released_hold_cannot_be_captured(
    session, funded, settlement_account, idem_key
):
    """The cancel-then-fill race, which is a real sequence and not a rare one."""
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    release_hold(session, hold=result.hold, idempotency_key=idem_key)

    with pytest.raises(HoldNotClaimable):
        capture_hold(
            session,
            hold=result.hold,
            settlement_account=settlement_account,
            idempotency_key=idem_key,
        )
    assert available_balance(session, funded.id) == 10_000_00


# ---------------------------------------------------------------------------
# The race
# ---------------------------------------------------------------------------


def test_concurrent_captures_settle_exactly_once(
    session, funded, settlement_account, idem_key
):
    """Eight workers, one hold, one fill.

    This is the test the whole ordering argument exists for. If the entries
    were posted before the hold was claimed, several of these would succeed,
    every transaction would balance perfectly, and the holding account would
    end up negative by a multiple of the fill — which no per-transaction check
    would ever notice.
    """
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    hold_id = result.hold.id
    settlement_id = settlement_account.id

    barrier = threading.Barrier(8)
    outcomes: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            with SessionLocal() as s:
                hold = s.get(Hold, hold_id)
                settlement = s.get(type(settlement_account), settlement_id)
                _, changed = capture_hold(
                    s,
                    hold=hold,
                    settlement_account=settlement,
                    idempotency_key=idem_key,
                    amount_minor=3_000_00,
                )
            with lock:
                outcomes.append(changed)
        except HoldNotClaimable:
            with lock:
                outcomes.append(False)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"
    assert sum(outcomes) == 1, f"{sum(outcomes)} captures claimed the hold"

    session.expire_all()
    assert held_balance(session, result.hold.held_account_id) == 0
    assert available_balance(session, funded.id) == 7_000_00


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def test_an_expired_hold_cannot_be_captured(
    session, funded, settlement_account, idem_key
):
    result = place_hold(
        session,
        idempotency_key=idem_key,
        cash_account=funded,
        amount_minor=3_000_00,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(HoldNotClaimable):
        capture_hold(
            session,
            hold=result.hold,
            settlement_account=settlement_account,
            idempotency_key=idem_key,
        )


def test_the_sweeper_returns_expired_money(session, funded, idem_key):
    """A stuck hold breaks no invariant. It is simply cash the customer owns
    and cannot use, against an order that no longer exists — so nothing else in
    the system will ever notice it."""
    place_hold(
        session,
        idempotency_key=idem_key,
        cash_account=funded,
        amount_minor=3_000_00,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert available_balance(session, funded.id) == 7_000_00

    released = expire_due_holds(session)

    assert len(released) == 1
    assert released[0].state == HoldState.expired
    assert available_balance(session, funded.id) == 10_000_00


def test_the_sweeper_leaves_live_holds_alone(session, funded, idem_key):
    place_hold(
        session,
        idempotency_key=idem_key,
        cash_account=funded,
        amount_minor=3_000_00,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert expire_due_holds(session) == []
    assert available_balance(session, funded.id) == 7_000_00


def test_a_hold_with_no_expiry_never_expires(session, funded, idem_key):
    place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=3_000_00
    )
    assert expire_due_holds(session) == []


# ---------------------------------------------------------------------------
# What the schema refuses on its own
# ---------------------------------------------------------------------------


def test_the_database_refuses_a_resolved_hold_with_no_resolution(
    session, funded, idem_key
):
    """The constraint that makes the claim-then-post ordering safe.

    Without it, a crash between marking a hold captured and recording which
    transaction did so would leave money gone from the holding account with
    nothing saying where — and the reconciliation job could report the
    discrepancy but never explain it.
    """
    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=1_000_00
    )
    from sqlalchemy import text

    # DatabaseError rather than IntegrityError: psycopg raises the latter and
    # pg8000 the former for the same refusal, and which one appears says
    # nothing about whether the constraint exists. IntegrityError and
    # InternalError are both subclasses, so this still catches them.
    with pytest.raises(DatabaseError):
        session.execute(
            text("UPDATE holds SET state = 'captured' WHERE id = :id"),
            {"id": result.hold.id},
        )
        session.commit()
    session.rollback()


def test_the_database_refuses_an_unknown_hold_state(session, funded, idem_key):
    from sqlalchemy import text

    result = place_hold(
        session, idempotency_key=idem_key, cash_account=funded, amount_minor=1_000_00
    )
    with pytest.raises(DatabaseError):
        session.execute(
            text("UPDATE holds SET state = 'banana' WHERE id = :id"),
            {"id": result.hold.id},
        )
        session.commit()
    session.rollback()


def test_concurrent_places_of_one_key_place_one_hold(session, funded, idem_key):
    """The race that only appeared once place_hold became atomic.

    The transaction row carries a unique idempotency key, so inserting it is
    itself a contention point -- and it is reached before any hold-level check,
    because the claim needs the transaction's id. Before this was handled, seven
    of eight callers got a raw IntegrityError out of what is simply a retry, and
    in production a duplicated request would have returned 500 rather than the
    hold it was asking about.
    """
    account_id = funded.id
    account_type = type(funded)

    barrier = threading.Barrier(8)
    created_flags: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            with SessionLocal() as s:
                account = s.get(account_type, account_id)
                result = place_hold(
                    s,
                    idempotency_key=idem_key,
                    cash_account=account,
                    amount_minor=2_000_00,
                )
            with lock:
                created_flags.append(result.created)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"
    assert sum(created_flags) == 1, f"{sum(created_flags)} callers placed a hold"

    session.expire_all()
    # One hold, so exactly one 2,000.00 left the spendable balance.
    assert available_balance(session, account_id) == 8_000_00
