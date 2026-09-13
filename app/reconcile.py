"""Proving the ledger is still what it claims to be.

    python -m app.reconcile

A fair question: the double-entry invariant is already a deferred constraint
trigger, held accounts are already tied to holds by a CHECK, and reversals are
already unique. What is left to reconcile?

Three things, and each is a real way a correct-looking database goes wrong.

CONSTRAINTS CAN BE TURNED OFF. `tests/conftest.py` disables the append-only
trigger to truncate between tests, because there is no other way to clean up a
table that refuses deletion. That is legitimate and it is also proof that the
capability exists: a migration, a data fix, a restore from a dump with
`session_replication_role = replica`, or anyone with the right grant can write
rows the triggers never saw. A check that only runs at write time cannot notice
that it did not run.

SOME INVARIANTS SPAN ROWS THAT NO CONSTRAINT CAN SEE TOGETHER. "The holding
account's balance equals the sum of the active holds against it" relates a SUM
over entries to a SUM over holds. No CHECK can express that, no trigger can
enforce it without serialising every write, and it is precisely the statement
that the holds table and the entries have not drifted apart. It is the only
check here that compares two independent representations of the same fact, so
it is the one that would actually catch a bug in `holds.py`.

AND A LEDGER CAN SATISFY EVERY CONSTRAINT WHILE BEING WRONG. A cash account
overdrawn to minus four lakh balances perfectly -- the money went somewhere, the
entries sum to zero, every transaction is valid. It is still a customer
spending money they do not have.

Each check returns the offending rows, not a boolean. "Reconciliation failed" at
three in the morning is not actionable; "account 41 is overdrawn by 4,00,000
paise and here are the three transactions that did it" is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Account, AccountType, Entry, Hold, HoldState, Transaction


@dataclass
class Finding:
    """One thing that is wrong, with enough detail to act on it."""

    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def add(self, check: str, detail: str) -> None:
        self.findings.append(Finding(check, detail))

    def summary(self) -> str:
        if self.clean:
            return (
                f"RECONCILED — {len(self.checks_run)} checks passed: "
                + ", ".join(self.checks_run)
            )
        lines = [f"{len(self.findings)} DISCREPANCIES across {len(self.checks_run)} checks:"]
        lines += [f"  {finding}" for finding in self.findings]
        return "\n".join(lines)


def every_transaction_balances(session: Session, report: Report) -> None:
    """The invariant the trigger enforces, verified independently.

    If this ever fails, the rows were written with the trigger disabled. That is
    worth knowing loudly, because every other number in the system is derived
    from these entries.
    """
    report.checks_run.append("transaction_balance")
    rows = session.execute(
        select(Entry.transaction_id, func.sum(Entry.amount_minor).label("total"))
        .group_by(Entry.transaction_id)
        .having(func.sum(Entry.amount_minor) != 0)
    ).all()
    for transaction_id, total in rows:
        report.add(
            "transaction_balance",
            f"transaction {transaction_id} sums to {total} paise, not 0",
        )


def the_books_balance(session: Session, report: Report) -> None:
    """Every paise that left an account arrived in another one."""
    report.checks_run.append("global_balance")
    total = session.scalar(
        select(func.coalesce(func.sum(Entry.amount_minor), 0))
    )
    if total != 0:
        report.add(
            "global_balance",
            f"all entries sum to {total} paise across the whole ledger, not 0",
        )


def every_transaction_has_two_legs(session: Session, report: Report) -> None:
    """A one-legged transaction sums to zero only if it is for zero paise.

    Which is to say it passes the balance trigger. It is still not a movement
    of value between two places, and it usually means a partially written
    transaction from code that committed too early.
    """
    report.checks_run.append("leg_count")
    rows = session.execute(
        select(Entry.transaction_id, func.count().label("legs"))
        .group_by(Entry.transaction_id)
        .having(func.count() < 2)
    ).all()
    for transaction_id, legs in rows:
        report.add("leg_count", f"transaction {transaction_id} has {legs} leg(s)")


def held_matches_active_holds(session: Session, report: Report) -> None:
    """The one check that compares two independent representations.

    Entries say how much is sitting in each holding account. The holds table
    says how much is claimed. A bug anywhere in `holds.py` -- a capture that
    posts twice, a release that forgets its legs, a claim that wins but does not
    commit -- shows up here as a difference, and nowhere else, because both
    sides of every one of those transactions balance perfectly on their own.
    """
    report.checks_run.append("held_vs_holds")

    entry_totals = dict(
        session.execute(
            select(Entry.account_id, func.coalesce(func.sum(Entry.amount_minor), 0))
            .join(Account, Account.id == Entry.account_id)
            .where(Account.type == AccountType.user_cash_held)
            .group_by(Entry.account_id)
        ).all()
    )
    claimed = dict(
        session.execute(
            select(Hold.held_account_id, func.coalesce(func.sum(Hold.amount_minor), 0))
            .where(Hold.state == HoldState.active)
            .group_by(Hold.held_account_id)
        ).all()
    )

    for account_id in set(entry_totals) | set(claimed):
        in_account = entry_totals.get(account_id, 0)
        against_holds = claimed.get(account_id, 0)
        if in_account != against_holds:
            report.add(
                "held_vs_holds",
                f"held account {account_id} holds {in_account} paise but "
                f"active holds claim {against_holds} "
                f"(difference {in_account - against_holds})",
            )


def no_overdrawn_cash(session: Session, report: Report) -> None:
    """A customer cannot spend money they do not have.

    Nothing in the schema prevents this: a negative balance is a perfectly
    valid set of balanced entries. It is enforced by a check at hold time,
    which under concurrency is advisory -- two simultaneous holds can each see
    enough available cash. This is where that gets caught.
    """
    report.checks_run.append("no_overdraft")
    rows = session.execute(
        select(Account.id, Account.owner_id, func.coalesce(func.sum(Entry.amount_minor), 0))
        .join(Entry, Entry.account_id == Account.id)
        .where(Account.type.in_([AccountType.user_cash, AccountType.user_cash_held]))
        .group_by(Account.id, Account.owner_id)
        .having(func.coalesce(func.sum(Entry.amount_minor), 0) < 0)
    ).all()
    for account_id, owner_id, balance in rows:
        report.add(
            "no_overdraft",
            f"account {account_id} (owner {owner_id}) is overdrawn by "
            f"{-balance} paise",
        )


def holds_resolve_consistently(session: Session, report: Report) -> None:
    """Resolved holds point at a real transaction; active ones point at none.

    The schema enforces the shape of this. What it cannot enforce is that the
    transaction pointed at actually exists and is the right kind, which is what
    a restore from a partial dump would break.
    """
    report.checks_run.append("hold_resolution")
    rows = session.scalars(select(Hold)).all()
    for hold in rows:
        terminal = hold.state != HoldState.active
        if terminal and hold.resolve_transaction_id is None:
            report.add(
                "hold_resolution",
                f"hold {hold.id} is {hold.state} with no resolving transaction",
            )
        if not terminal and hold.resolve_transaction_id is not None:
            report.add(
                "hold_resolution",
                f"hold {hold.id} is active but points at transaction "
                f"{hold.resolve_transaction_id}",
            )
        if hold.resolve_transaction_id is not None:
            txn = session.get(Transaction, hold.resolve_transaction_id)
            if txn is None:
                report.add(
                    "hold_resolution",
                    f"hold {hold.id} points at transaction "
                    f"{hold.resolve_transaction_id}, which does not exist",
                )


def reversals_mirror_their_originals(session: Session, report: Report) -> None:
    """A reversal must undo exactly what it claims to undo.

    Summing the original and its reversal together has to give zero per
    account. A reversal that negates the wrong legs still balances, still
    passes every constraint, and leaves two accounts wrong in opposite
    directions.
    """
    report.checks_run.append("reversal_mirrors")
    reversals = session.scalars(
        select(Transaction).where(Transaction.reverses_transaction_id.is_not(None))
    ).all()

    for reversal in reversals:
        pair = session.execute(
            select(Entry.account_id, func.sum(Entry.amount_minor))
            .where(
                Entry.transaction_id.in_(
                    [reversal.id, reversal.reverses_transaction_id]
                )
            )
            .group_by(Entry.account_id)
            .having(func.sum(Entry.amount_minor) != 0)
        ).all()
        for account_id, residue in pair:
            report.add(
                "reversal_mirrors",
                f"transaction {reversal.id} reverses "
                f"{reversal.reverses_transaction_id} but leaves {residue} paise "
                f"on account {account_id}",
            )


CHECKS = (
    every_transaction_balances,
    the_books_balance,
    every_transaction_has_two_legs,
    held_matches_active_holds,
    no_overdrawn_cash,
    holds_resolve_consistently,
    reversals_mirror_their_originals,
)


def reconcile(session: Session) -> Report:
    report = Report()
    for check in CHECKS:
        check(session, report)
    return report


def main() -> None:
    from app.db import SessionLocal

    with SessionLocal() as session:
        report = reconcile(session)

    print(report.summary())
    # Non-zero exit so this can be a cron job or a CI step without anyone
    # having to parse the output to find out whether it passed.
    sys.exit(0 if report.clean else 1)


if __name__ == "__main__":
    main()
