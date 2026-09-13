"""Reconciliation, tested by breaking the ledger on purpose.

A reconciliation job that has only ever been run against correct data is not
known to work. So each check here is given damage of exactly the kind it exists
to find, and is required to name it.

The damage is inflicted the same way real damage arrives: with the triggers
switched off. That is not a contrived hole -- `conftest.py` does it to truncate
between tests, a migration can do it, `session_replication_role = replica` does
it for the whole session, and a restore from a logical dump does it by default.
The reason to have a reconciliation job at all is that write-time checks cannot
see the writes that happened while they were not looking.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from sqlalchemy import text

from app.holds import capture_hold, place_hold, release_hold
from app.ledger import Leg, post_transaction, reverse_transaction
from app.models import Hold, HoldState
from app.reconcile import (
    Report,
    every_transaction_balances,
    every_transaction_has_two_legs,
    held_matches_active_holds,
    holds_resolve_consistently,
    no_overdrawn_cash,
    reconcile,
    reversals_mirror_their_originals,
    the_books_balance,
)


def fund(session, cash_account, settlement_account, paise=10_000_00):
    post_transaction(
        session,
        idempotency_key=f"fund-{uuid.uuid4()}",
        kind="deposit",
        legs=[Leg(cash_account.id, paise), Leg(settlement_account.id, -paise)],
    )


#: The two triggers guarding `entries`. Disabled as table owner rather than
#: with `session_replication_role`, which Neon does not grant -- a useful
#: reminder that the privilege needed to bypass these checks is the ordinary
#: one every migration already has.
ENTRY_TRIGGERS = ("entries_balance_check", "entries_no_mutation")


def without_triggers(session, statement: str, **params) -> None:
    """Write rows the constraints never saw. See the module docstring."""
    for trigger in ENTRY_TRIGGERS:
        session.execute(text(f"ALTER TABLE entries DISABLE TRIGGER {trigger}"))
    session.execute(text(statement), params)
    for trigger in ENTRY_TRIGGERS:
        session.execute(text(f"ALTER TABLE entries ENABLE TRIGGER {trigger}"))
    session.commit()
    # Raw SQL is invisible to the identity map, and this session is built with
    # expire_on_commit=False. Without this the checks below would read the
    # pre-damage objects and pass. Production never hits this -- reconcile runs
    # in a session of its own -- but a test that passes for that reason would
    # be worse than no test.
    session.expire_all()


@contextmanager
def without_constraint(session, table: str, name: str):
    """Drop a CHECK constraint, damage the data, then put the constraint back.

    Restored as NOT VALID, so the row just written does not block its own
    restoration. Future writes are enforced again, which is what matters for
    the tests that follow.
    """
    definition = session.execute(
        text("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = :n"),
        {"n": name},
    ).scalar_one()
    session.execute(text(f"ALTER TABLE {table} DROP CONSTRAINT {name}"))
    session.commit()
    try:
        yield
    finally:
        session.rollback()
        session.execute(
            text(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition} NOT VALID")
        )
        session.commit()


def damage_hold(session, statement: str, **params) -> None:
    """Mutate a hold past the constraint that would normally refuse it."""
    with without_constraint(session, "holds", "holds_resolution_consistent"):
        session.execute(text(statement), params)
        session.commit()
    session.expire_all()


@pytest.fixture
def busy(session, cash_account, settlement_account):
    """A ledger with a bit of everything in it, all of it correct."""
    fund(session, cash_account, settlement_account)

    kept = place_hold(
        session, idempotency_key="keep", cash_account=cash_account, amount_minor=1_000_00
    )
    captured = place_hold(
        session, idempotency_key="cap", cash_account=cash_account, amount_minor=2_000_00
    )
    capture_hold(
        session,
        hold=captured.hold,
        settlement_account=settlement_account,
        idempotency_key="cap",
        amount_minor=1_500_00,
    )
    released = place_hold(
        session, idempotency_key="rel", cash_account=cash_account, amount_minor=500_00
    )
    release_hold(session, hold=released.hold, idempotency_key="rel")

    txn, _ = post_transaction(
        session,
        idempotency_key="fee",
        kind="fee",
        legs=[Leg(cash_account.id, -100_00), Leg(settlement_account.id, 100_00)],
    )
    reverse_transaction(session, transaction_id=txn.id, idempotency_key="fee-rev")
    return kept.hold


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def test_a_healthy_ledger_reconciles(session, busy):
    report = reconcile(session)
    assert report.clean, report.summary()
    assert len(report.checks_run) == 7


def test_an_empty_ledger_reconciles(session):
    assert reconcile(session).clean


def test_the_summary_names_the_checks_that_ran(session, busy):
    summary = reconcile(session).summary()
    assert "RECONCILED" in summary
    assert "held_vs_holds" in summary


# ---------------------------------------------------------------------------
# Broken, one way at a time
# ---------------------------------------------------------------------------


def test_an_unbalanced_transaction_is_caught(session, busy, cash_account):
    """Rows written while the deferred balance trigger was not watching."""
    without_triggers(
        session,
        "INSERT INTO transactions (idempotency_key, kind) "
        "VALUES ('bad', 'smuggled') RETURNING id",
    )
    txn_id = session.execute(
        text("SELECT id FROM transactions WHERE idempotency_key = 'bad'")
    ).scalar_one()
    without_triggers(
        session,
        "INSERT INTO entries (transaction_id, account_id, amount_minor) "
        "VALUES (:t, :a, 777)",
        t=txn_id,
        a=cash_account.id,
    )

    report = Report()
    every_transaction_balances(session, report)

    assert not report.clean
    assert str(txn_id) in report.findings[0].detail
    assert "777" in report.findings[0].detail


def test_the_books_not_balancing_is_caught(session, busy, cash_account):
    without_triggers(
        session,
        "INSERT INTO transactions (idempotency_key, kind) VALUES ('bad2', 'smuggled')",
    )
    txn_id = session.execute(
        text("SELECT id FROM transactions WHERE idempotency_key = 'bad2'")
    ).scalar_one()
    without_triggers(
        session,
        "INSERT INTO entries (transaction_id, account_id, amount_minor) "
        "VALUES (:t, :a, 500)",
        t=txn_id,
        a=cash_account.id,
    )

    report = Report()
    the_books_balance(session, report)
    assert not report.clean
    assert "500" in report.findings[0].detail


def test_a_one_legged_transaction_is_caught(session, busy, cash_account):
    """Sums to zero, so the balance trigger is content. Still not a movement."""
    without_triggers(
        session,
        "INSERT INTO transactions (idempotency_key, kind) VALUES ('lonely', 'half')",
    )
    txn_id = session.execute(
        text("SELECT id FROM transactions WHERE idempotency_key = 'lonely'")
    ).scalar_one()
    without_triggers(
        session,
        "INSERT INTO entries (transaction_id, account_id, amount_minor) "
        "VALUES (:t, :a, 0)",
        t=txn_id,
        a=cash_account.id,
    )

    report = Report()
    every_transaction_balances(session, report)
    assert report.clean, "a zero-value single leg balances; that is the point"

    report = Report()
    every_transaction_has_two_legs(session, report)
    assert not report.clean
    assert "1 leg" in report.findings[0].detail


def test_a_hold_that_lost_track_of_its_money_is_caught(session, busy):
    """The check that compares two independent representations.

    Here a hold is silently marked released without its legs ever being
    posted -- exactly what a capture that claimed the row and then failed to
    commit its entries would leave behind. Every transaction in the database
    still balances; only the cross-check sees it.
    """
    damage_hold(
        session,
        "UPDATE holds SET state = 'released', resolved_at = now(), "
        "resolve_transaction_id = place_transaction_id WHERE id = :id",
        id=busy.id,
    )

    report = Report()
    held_matches_active_holds(session, report)

    assert not report.clean
    assert "100000" in report.findings[0].detail  # the 1,000.00 still sitting there


def test_an_overdrawn_customer_is_caught(session, cash_account, settlement_account):
    """A perfectly balanced set of entries that should never have happened."""
    post_transaction(
        session,
        idempotency_key="overspend",
        kind="withdrawal",
        legs=[Leg(cash_account.id, -4_00_000), Leg(settlement_account.id, 4_00_000)],
    )

    report = Report()
    no_overdrawn_cash(session, report)

    assert not report.clean
    assert "overdrawn by 400000" in report.findings[0].detail


def test_a_house_account_may_go_negative(session, cash_account, settlement_account):
    """The house settlement account is negative by construction whenever a
    customer holds money. Flagging it would make the check useless."""
    fund(session, cash_account, settlement_account)
    report = Report()
    no_overdrawn_cash(session, report)
    assert report.clean


def test_a_hold_resolved_into_nowhere_is_caught(session, busy):
    damage_hold(
        session,
        "UPDATE holds SET state = 'captured', resolved_at = now() WHERE id = :id",
        id=busy.id,
    )

    report = Report()
    holds_resolve_consistently(session, report)
    assert not report.clean
    assert "no resolving transaction" in report.findings[0].detail


def test_an_active_hold_pointing_at_a_transaction_is_caught(session, busy):
    damage_hold(
        session,
        "UPDATE holds SET resolve_transaction_id = place_transaction_id "
        "WHERE id = :id",
        id=busy.id,
    )
    report = Report()
    holds_resolve_consistently(session, report)
    assert not report.clean
    assert "active but points at" in report.findings[0].detail


def test_a_reversal_that_does_not_mirror_is_caught(
    session, cash_account, settlement_account
):
    """A reversal with the wrong legs balances and passes every constraint. It
    just leaves two accounts wrong in opposite directions."""
    fund(session, cash_account, settlement_account)
    original, _ = post_transaction(
        session,
        idempotency_key="orig",
        kind="fee",
        legs=[Leg(cash_account.id, -100_00), Leg(settlement_account.id, 100_00)],
    )
    without_triggers(
        session,
        "INSERT INTO transactions (idempotency_key, kind, reverses_transaction_id) "
        "VALUES ('wrong-rev', 'reversal:fee', :orig)",
        orig=original.id,
    )
    bad_id = session.execute(
        text("SELECT id FROM transactions WHERE idempotency_key = 'wrong-rev'")
    ).scalar_one()
    # Reverses the right accounts for the wrong amount.
    without_triggers(
        session,
        "INSERT INTO entries (transaction_id, account_id, amount_minor) "
        "VALUES (:t, :cash, 50000), (:t, :house, -50000)",
        t=bad_id,
        cash=cash_account.id,
        house=settlement_account.id,
    )

    report = Report()
    reversals_mirror_their_originals(session, report)

    assert not report.clean
    assert len(report.findings) == 2  # both accounts left wrong
    assert "leaves" in report.findings[0].detail


def test_every_finding_names_something_actionable(session, busy, cash_account):
    """A reconciliation failure at 3am has to say WHAT, not just THAT."""
    post_transaction(
        session,
        idempotency_key="overspend",
        kind="withdrawal",
        legs=[Leg(cash_account.id, -50_00_000), Leg(busy.held_account_id, 50_00_000)],
    )
    report = reconcile(session)

    assert not report.clean
    for finding in report.findings:
        assert finding.check
        assert any(ch.isdigit() for ch in finding.detail), finding.detail
    assert "DISCREPANCIES" in report.summary()
