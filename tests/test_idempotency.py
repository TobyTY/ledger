"""Idempotency: one key posts exactly once, however hard you push on it."""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal
from app.ledger import Leg, Unbalanced, account_balance, post_transaction
from app.models import Entry, Transaction

DEPOSIT = 100_000  # ₹1,000.00 in paise


def _legs(cash_id: int, settlement_id: int) -> list[Leg]:
    return [Leg(cash_id, DEPOSIT), Leg(settlement_id, -DEPOSIT)]


def test_same_key_posts_once(session, cash_account, settlement_account, idem_key):
    legs = _legs(cash_account.id, settlement_account.id)

    first, created_first = post_transaction(
        session, idempotency_key=idem_key, kind="deposit", legs=legs
    )
    second, created_second = post_transaction(
        session, idempotency_key=idem_key, kind="deposit", legs=legs
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id

    # The retry moved no money.
    assert account_balance(session, cash_account.id) == DEPOSIT


def test_different_keys_post_separately(session, cash_account, settlement_account):
    legs = _legs(cash_account.id, settlement_account.id)

    a, _ = post_transaction(
        session, idempotency_key=f"k-{uuid.uuid4()}", kind="deposit", legs=legs
    )
    b, _ = post_transaction(
        session, idempotency_key=f"k-{uuid.uuid4()}", kind="deposit", legs=legs
    )

    assert a.id != b.id
    assert account_balance(session, cash_account.id) == DEPOSIT * 2


def test_unbalanced_legs_are_rejected(session, cash_account, settlement_account, idem_key):
    with pytest.raises(Unbalanced, match="must sum to 0"):
        post_transaction(
            session,
            idempotency_key=idem_key,
            kind="deposit",
            legs=[Leg(cash_account.id, DEPOSIT), Leg(settlement_account.id, -1)],
        )


def test_database_rejects_unbalanced_even_without_the_app_guard(
    session, cash_account, settlement_account, idem_key, monkeypatch
):
    """The database is the guarantee; the application check is a courtesy.

    Disable the application-level balance check and the write must still fail --
    and it must fail as a check violation that propagates, not get swallowed by
    the duplicate-key path as "already posted".
    """
    monkeypatch.setattr("app.ledger._assert_balanced", lambda legs: None)

    with pytest.raises(IntegrityError) as exc_info:
        post_transaction(
            session,
            idempotency_key=idem_key,
            kind="deposit",
            legs=[Leg(cash_account.id, DEPOSIT), Leg(settlement_account.id, -1)],
        )
    assert "does not balance" in str(exc_info.value)

    session.rollback()
    assert session.scalar(
        select(func.count()).select_from(Transaction).where(
            Transaction.idempotency_key == idem_key
        )
    ) == 0


def test_concurrent_duplicate_keys_post_exactly_once(
    session, cash_account, settlement_account
):
    """Eight callers, one key, released simultaneously.

    Postgres serialises them on the unique index: the losers block until the
    winner commits, then raise 23505, roll back, and read the winner's row. The
    assertion that matters is not just "one transaction exists" but that every
    caller was handed the same one -- a client that retried must never be told
    its money went somewhere it did not.
    """
    cash_id, settlement_id = cash_account.id, settlement_account.id
    key = f"race-{uuid.uuid4()}"
    workers = 8

    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    results: list[tuple[int, bool]] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            with SessionLocal() as s:
                barrier.wait(timeout=30)  # maximise the overlap
                txn, created = post_transaction(
                    s,
                    idempotency_key=key,
                    kind="deposit",
                    legs=_legs(cash_id, settlement_id),
                )
                with lock:
                    results.append((txn.id, created))
        except Exception as exc:  # noqa: BLE001 - surfaced in the assertion below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"workers raised: {errors!r}"
    assert len(results) == workers

    transaction_ids = {txn_id for txn_id, _ in results}
    assert len(transaction_ids) == 1, "every caller must receive the same transaction"
    assert sum(created for _, created in results) == 1, "exactly one caller created it"

    session.rollback()  # fresh snapshot
    winner_id = transaction_ids.pop()
    assert session.scalar(
        select(func.count()).select_from(Transaction).where(
            Transaction.idempotency_key == key
        )
    ) == 1
    assert session.scalar(
        select(func.count()).select_from(Entry).where(Entry.transaction_id == winner_id)
    ) == 2
    assert account_balance(session, cash_id) == DEPOSIT
