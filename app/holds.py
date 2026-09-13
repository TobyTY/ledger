"""Placing, capturing and releasing holds.

A hold is what stands between "the customer placed an order" and "the customer
spent money". Three operations, and the whole difficulty is in the second:

    place    cash -> held           the order was accepted
    capture  held -> settlement     the order filled, wholly or partly
    release  held -> cash           the order was cancelled, or expired

WHY A CAPTURE IS HARD. An exchange fill callback is not delivered once. It is
delivered at least once, sometimes twice, occasionally after the customer has
already cancelled, and two of them can arrive simultaneously on different
workers. Every one of those paths must end with the money having moved exactly
once. So the state transition is the thing that is made atomic, and the entries
follow it:

    UPDATE holds
       SET state = 'captured', resolve_transaction_id = :txn, resolved_at = now()
     WHERE id = :id AND state = 'active'

The row count decides. One means this caller claimed the hold and may post the
legs; zero means somebody else already did, or it expired, and this caller must
post nothing. Two concurrent captures serialise on the row lock, and Postgres
re-evaluates the WHERE clause against the committed version -- so the second one
matches no rows rather than overwriting the first.

The transaction row is created BEFORE the claim, so its id can go into that same
UPDATE, and the legs are written after it. All of it commits once. Splitting the
state change from the resolution -- mark captured, commit, then record which
transaction did it -- would leave a window where a crash strands a hold saying
"captured" with no record of where the money went, which the schema refuses
outright and which no amount of care in application code would prevent.

The opposite order -- post the legs, then mark the hold -- looks equivalent and
is not. Two callers would both post, and the holding account would go negative
by exactly one fill, which no per-transaction check would notice because each
transaction balances perfectly on its own.

PARTIAL FILLS. A capture of less than the full hold releases the remainder in
the SAME transaction, as a third leg. Leaving the difference stranded in the
holding account would be money the customer owns, cannot spend, and has nothing
open against -- invisible to every balance an account holder is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from sqlalchemy.exc import IntegrityError

from app.ledger import Leg, add_legs, begin_transaction, is_unique_violation
from app.models import Account, AccountType, Entry, Hold, HoldState


class HoldError(Exception):
    """A hold operation that cannot be honoured."""


class InsufficientFunds(HoldError):
    """Not enough spendable cash to place this hold."""


class HoldNotClaimable(HoldError):
    """The hold was already resolved, or has expired.

    Raised by both capture and release, and deliberately the same exception for
    both causes: from the caller's side "somebody else got there first" and "it
    timed out" require the identical response, which is to stop.
    """


def _open_transaction(
    session: Session, *, key: str, kind: str
) -> tuple[object | None, bool]:
    """Start the resolving transaction, or report that someone else already has.

    The transaction row carries a unique idempotency key, so inserting it is
    itself a contention point -- and it is reached BEFORE the hold claim,
    because the claim needs the transaction's id. Concurrent callers therefore
    collide here rather than on the hold, and the loser has to recognise that
    for what it is.

    Postgres makes this clean: the losing INSERT blocks on the unique index
    until the winner's transaction ends. A 23505 is only raised once the winner
    has COMMITTED, so by the time this returns False the hold is already
    resolved and readable.

    Returns (transaction, True) for the winner, (None, False) for everyone else.
    """
    try:
        return begin_transaction(session, idempotency_key=key, kind=kind), True
    except IntegrityError as exc:
        session.rollback()
        if not is_unique_violation(exc):
            raise
        return None, False


@dataclass(frozen=True)
class HoldResult:
    hold: Hold
    created: bool


def available_balance(session: Session, cash_account_id: int) -> int:
    """Spendable paise: the cash account's own balance.

    Held funds are not subtracted here because they are not here -- placing a
    hold moved them to another account. That is the entire reason for modelling
    a hold as a transfer, and it is why this function needs no knowledge of
    holds at all.
    """
    return session.scalar(
        select(func.coalesce(func.sum(Entry.amount_minor), 0)).where(
            Entry.account_id == cash_account_id
        )
    )


def held_balance(session: Session, held_account_id: int) -> int:
    return session.scalar(
        select(func.coalesce(func.sum(Entry.amount_minor), 0)).where(
            Entry.account_id == held_account_id
        )
    )


def held_account_for(session: Session, cash_account: Account) -> Account:
    """The customer's holding account, created on first use.

    One per cash account. Two would make the reconciliation check "held balance
    equals the sum of active holds" ambiguous about which balance it means.
    """
    existing = session.scalar(
        select(Account).where(
            Account.type == AccountType.user_cash_held,
            Account.owner_id == cash_account.owner_id,
            Account.currency == cash_account.currency,
        )
    )
    if existing is not None:
        return existing

    account = Account(
        type=AccountType.user_cash_held,
        owner_id=cash_account.owner_id,
        currency=cash_account.currency,
    )
    session.add(account)
    session.commit()
    return account


def _find_hold(session: Session, idempotency_key: str) -> Hold | None:
    return session.scalar(
        select(Hold).where(Hold.idempotency_key == idempotency_key)
    )


def place_hold(
    session: Session,
    *,
    idempotency_key: str,
    cash_account: Account,
    amount_minor: int,
    expires_at: datetime | None = None,
) -> HoldResult:
    """Move `amount_minor` out of spendable cash and record the claim."""
    if amount_minor <= 0:
        raise HoldError(f"hold amount must be positive, got {amount_minor}")

    existing = _find_hold(session, idempotency_key)
    if existing is not None:
        return HoldResult(existing, created=False)

    # Checked before posting, and re-checked by the reconciliation job rather
    # than by a constraint: a per-row CHECK cannot see an account's balance, and
    # a trigger that sums entries on every insert would serialise every write to
    # a busy account. The cost of that choice is that this check is advisory
    # under concurrency, which is exactly what `no_overdrawn_cash` exists to
    # catch.
    available = available_balance(session, cash_account.id)
    if available < amount_minor:
        raise InsufficientFunds(
            f"account {cash_account.id} has {available} paise available, "
            f"hold needs {amount_minor}"
        )

    held_account = held_account_for(session, cash_account)

    # The legs and the hold row commit together, for the same reason a capture
    # claims before it posts. Moving the money first and recording the claim
    # afterwards leaves a window where a crash strands cash in the holding
    # account with no hold against it -- which `held_vs_holds` would report as
    # a discrepancy, correctly, and which nobody could then explain.
    txn, won = _open_transaction(
        session, key=f"hold:place:{idempotency_key}", kind="hold_place"
    )
    if not won:
        # A concurrent caller placed this exact hold. Theirs is the answer.
        return HoldResult(_find_hold(session, idempotency_key), created=False)

    add_legs(
        session,
        txn,
        [Leg(cash_account.id, -amount_minor), Leg(held_account.id, amount_minor)],
    )
    session.add(
        Hold(
            idempotency_key=idempotency_key,
            cash_account_id=cash_account.id,
            held_account_id=held_account.id,
            amount_minor=amount_minor,
            state=HoldState.active,
            place_transaction_id=txn.id,
            expires_at=expires_at,
        )
    )
    session.commit()

    hold = _find_hold(session, idempotency_key)
    return HoldResult(hold, created=True)


def _claim(
    session: Session,
    hold_id: int,
    new_state: HoldState,
    *,
    resolve_transaction_id: int,
    captured_minor: int,
    allow_expired: bool = False,
) -> bool:
    """Atomically resolve a hold. True if this caller won it.

    State and resolution move in ONE statement. Setting the state first and the
    resolving transaction afterwards would mean a commit could land between
    them, leaving a hold marked captured with nothing recording where the money
    went -- which the schema forbids outright, and which would otherwise be
    reachable simply by the process dying at the wrong moment.

    Expiry is part of the WHERE clause rather than a read followed by a check,
    so a hold cannot expire in the gap between the two. `allow_expired` exists
    for the sweeper, which is claiming holds precisely BECAUSE they expired.
    """
    conditions = [Hold.id == hold_id, Hold.state == HoldState.active]
    if not allow_expired:
        conditions.append(
            (Hold.expires_at.is_(None)) | (Hold.expires_at > func.now())
        )

    result = session.execute(
        update(Hold)
        .where(*conditions)
        .values(
            state=new_state,
            resolve_transaction_id=resolve_transaction_id,
            captured_minor=captured_minor,
            resolved_at=datetime.now(timezone.utc),
        )
    )
    return result.rowcount == 1


def capture_hold(
    session: Session,
    *,
    hold: Hold,
    settlement_account: Account,
    idempotency_key: str,
    amount_minor: int | None = None,
) -> tuple[Hold, bool]:
    """Convert a hold, wholly or partly, into a settled movement.

    `amount_minor` defaults to the full hold. Anything left over is released
    back to spendable cash in the same transaction, because a remainder left in
    the holding account is money the customer owns, cannot spend, and has
    nothing open against.
    """
    session.refresh(hold)
    capture = hold.amount_minor if amount_minor is None else amount_minor

    if capture < 0 or capture > hold.amount_minor:
        raise HoldError(
            f"cannot capture {capture} paise from a hold of {hold.amount_minor}"
        )

    if _already_captured(hold, capture):
        return hold, False

    txn, won = _open_transaction(
        session, key=f"hold:capture:{idempotency_key}", kind="hold_capture"
    )
    if not won:
        session.refresh(hold)
        if _already_captured(hold, capture):
            return hold, False
        raise HoldNotClaimable(
            f"hold {hold.id} was captured concurrently for "
            f"{hold.captured_minor}, not {capture}"
        )

    if not _claim(
        session,
        hold.id,
        HoldState.captured,
        resolve_transaction_id=txn.id,
        captured_minor=capture,
    ):
        # Nothing has been written that anyone can see: the transaction row
        # above dies with this rollback.
        session.rollback()
        session.refresh(hold)
        if _already_captured(hold, capture):
            return hold, False
        raise HoldNotClaimable(
            f"hold {hold.id} is {hold.state} and cannot be captured"
        )

    legs = [Leg(hold.held_account_id, -hold.amount_minor)]
    if capture:
        legs.append(Leg(settlement_account.id, capture))
    remainder = hold.amount_minor - capture
    if remainder:
        legs.append(Leg(hold.cash_account_id, remainder))

    add_legs(session, txn, legs)
    session.commit()
    session.refresh(hold)
    return hold, True


def _already_captured(hold: Hold, capture: int) -> bool:
    """Is this the same capture arriving again?

    Fill callbacks are delivered at least once, so a repeat is the ordinary
    case rather than an error. A repeat for a DIFFERENT amount is not, and
    falls through to be refused.
    """
    return hold.state == HoldState.captured and hold.captured_minor == capture


def release_hold(
    session: Session, *, hold: Hold, idempotency_key: str, expired: bool = False
) -> tuple[Hold, bool]:
    """Return a hold's funds to spendable cash."""
    session.refresh(hold)
    target = HoldState.expired if expired else HoldState.released

    if hold.state in (HoldState.released, HoldState.expired):
        return hold, False

    txn, won = _open_transaction(
        session, key=f"hold:release:{idempotency_key}", kind="hold_release"
    )
    if not won:
        session.refresh(hold)
        if hold.state in (HoldState.released, HoldState.expired):
            return hold, False
        raise HoldNotClaimable(
            f"hold {hold.id} is {hold.state} and cannot be released"
        )

    if not _claim(
        session,
        hold.id,
        target,
        resolve_transaction_id=txn.id,
        captured_minor=0,
        allow_expired=expired,
    ):
        session.rollback()
        session.refresh(hold)
        if hold.state in (HoldState.released, HoldState.expired):
            return hold, False
        raise HoldNotClaimable(
            f"hold {hold.id} is {hold.state} and cannot be released"
        )

    add_legs(
        session,
        txn,
        [
            Leg(hold.held_account_id, -hold.amount_minor),
            Leg(hold.cash_account_id, hold.amount_minor),
        ],
    )
    session.commit()
    session.refresh(hold)
    return hold, True


def expire_due_holds(session: Session) -> list[Hold]:
    """Release every hold whose expiry has passed.

    Run on a schedule. An expired hold left active is cash the customer owns
    and cannot use, against an order that no longer exists -- and nothing else
    in the system will ever notice, because a stuck hold breaks no invariant.
    It is simply wrong.
    """
    due = session.scalars(
        select(Hold).where(
            Hold.state == HoldState.active,
            Hold.expires_at.is_not(None),
            Hold.expires_at <= func.now(),
        )
    ).all()

    released = []
    for hold in due:
        # The claim inside release_hold re-checks expiry against now(), so a
        # hold captured between this query and the update is skipped rather
        # than double-resolved.
        try:
            resolved, changed = release_hold(
                session,
                hold=hold,
                idempotency_key=f"expire:{hold.id}",
                expired=True,
            )
        except HoldNotClaimable:
            continue
        if changed:
            released.append(resolved)
    return released
