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


class ReversalRefused(ValueError):
    """This transaction must not be reversed through the generic path."""


@dataclass(frozen=True)
class Leg:
    """One side of a movement. Debits positive, credits negative, in paise."""

    account_id: int
    amount_minor: int


def _sqlstate(exc: Exception) -> str | None:
    """The five-character SQLSTATE, whichever driver raised the error.

    Driver-agnostic on purpose, and not for portability points. Every decision
    in this module turns on the difference between 23505 and 23514, and each
    driver exposes that code somewhere different:

        psycopg3   exc.orig.sqlstate
        psycopg2   exc.orig.pgcode
        pg8000     exc.orig.args[0]["C"]   -- a dict of the Postgres wire
                                              protocol's single-letter error
                                              fields; "C" is the SQLSTATE

    A version of this function that only knew psycopg returned None under
    pg8000, which made `is_unique_violation` answer False to every question.
    That does not raise anything. It just stops the idempotency race from being
    recognised, so the loser of a race reports a failure instead of returning
    the winner's transaction -- a silent behaviour change that no test of the
    happy path would notice.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return None

    code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    if code:
        return str(code)

    args = getattr(orig, "args", ())
    if args and isinstance(args[0], dict):
        return args[0].get("C")
    return None


def is_unique_violation(exc: IntegrityError) -> bool:
    """23505 specifically, never IntegrityError generally.

    Every caller that swallows a constraint failure has to make this
    distinction. A unique violation means somebody else got there first, which
    is recoverable. A check violation means the write was wrong, which is not,
    and treating the two alike is how an unbalanced transaction gets reported
    as a successful retry.
    """
    return _sqlstate(exc) == UNIQUE_VIOLATION


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


def begin_transaction(
    session: Session,
    *,
    idempotency_key: str,
    kind: str,
    reverses_transaction_id: int | None = None,
) -> Transaction:
    """Insert the transaction row and flush, so its id exists. No commit.

    Separated out because some callers need the id BEFORE the legs are written.
    A hold capture is the reason: it has to claim the hold and record which
    transaction resolved it in a single UPDATE, and it cannot do that without
    an id to point at. Committing in between would leave the hold briefly
    marked captured with no resolving transaction -- a state the schema
    forbids, and rightly.
    """
    txn = Transaction(
        idempotency_key=idempotency_key,
        kind=kind,
        reverses_transaction_id=reverses_transaction_id,
    )
    session.add(txn)
    # The unique index on idempotency_key is checked here, immediately.
    session.flush()
    return txn


def add_legs(session: Session, txn: Transaction, legs: list[Leg]) -> None:
    """Attach entries to an uncommitted transaction.

    The balance trigger does not fire here. It is deferred to COMMIT, which is
    what lets a two-leg transaction exist half-written inside a transaction
    without being rejected.
    """
    for leg in legs:
        session.add(
            Entry(
                transaction_id=txn.id,
                account_id=leg.account_id,
                amount_minor=leg.amount_minor,
            )
        )


def post_transaction(
    session: Session,
    *,
    idempotency_key: str,
    kind: str,
    legs: list[Leg],
    reverses_transaction_id: int | None = None,
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

    try:
        txn = begin_transaction(
            session,
            idempotency_key=idempotency_key,
            kind=kind,
            reverses_transaction_id=reverses_transaction_id,
        )
        add_legs(session, txn, legs)
        # The balance trigger is DEFERRABLE INITIALLY DEFERRED, so an unbalanced
        # transaction surfaces here at COMMIT, not at the flush inside
        # begin_transaction.
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


def reverse_transaction(
    session: Session,
    *,
    transaction_id: int,
    idempotency_key: str,
    reason: str = "",
) -> tuple[Transaction, bool]:
    """Undo a transaction by posting its mirror image.

    Nothing is edited or deleted -- entries are append-only, so a correction is
    a new transaction whose legs are the originals negated. The original stays
    in history, which is what keeps any account's balance reconstructible as of
    any past moment.

    ONE REVERSAL PER TRANSACTION, and that is enforced by a unique index on
    `reverses_transaction_id` rather than by checking first. Two concurrent
    reversal requests both pass a check-then-insert; only one survives a unique
    constraint. The failure mode being prevented is quiet: each reversal
    balances perfectly on its own, so a per-transaction audit sees nothing
    wrong, and the account is simply over-credited by the original amount.

    A hold's own transactions are refused. Reversing a hold placement directly
    would return the funds to spendable cash while the hold row still says
    'active' -- the money would be both spendable and claimed, and the
    reconciliation job would report a discrepancy it could not explain. Holds
    are undone through `release_hold`, which moves the state machine with them.
    """
    from app.models import Hold  # local import: holds are a layer above legs

    original = session.get(Transaction, transaction_id)
    if original is None:
        raise ValueError(f"transaction {transaction_id} does not exist")

    if original.kind.startswith("hold_"):
        raise ReversalRefused(
            f"transaction {transaction_id} is part of a hold lifecycle "
            f"({original.kind}); release the hold instead"
        )

    existing = session.scalar(
        select(Transaction).where(
            Transaction.reverses_transaction_id == transaction_id
        )
    )
    if existing is not None:
        return existing, False

    legs = [
        Leg(entry.account_id, -entry.amount_minor)
        for entry in session.scalars(
            select(Entry).where(Entry.transaction_id == transaction_id)
        )
    ]
    if not legs:
        raise ValueError(f"transaction {transaction_id} has no entries to reverse")

    try:
        return post_transaction(
            session,
            idempotency_key=idempotency_key,
            kind=f"reversal:{original.kind}" + (f":{reason}" if reason else ""),
            legs=legs,
            reverses_transaction_id=transaction_id,
        )
    except IntegrityError as exc:
        # Lost the race for the unique index on reverses_transaction_id. The
        # winner's reversal is the correct answer for this caller too.
        session.rollback()
        if _sqlstate(exc) != UNIQUE_VIOLATION:
            raise
        winner = session.scalar(
            select(Transaction).where(
                Transaction.reverses_transaction_id == transaction_id
            )
        )
        if winner is None:
            raise
        return winner, False
