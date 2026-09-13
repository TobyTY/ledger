"""HTTP surface.

Idempotency is carried in the ``Idempotency-Key`` header rather than the body,
matching the convention Stripe popularised -- it is a property of the request,
not of the money being moved.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, Field, StrictInt, StrictStr
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.holds import (
    HoldError,
    HoldNotClaimable,
    InsufficientFunds,
    available_balance,
    capture_hold,
    place_hold,
    release_hold,
)
from app.ledger import (
    Leg,
    ReversalRefused,
    Unbalanced,
    account_balance,
    post_transaction,
    reverse_transaction,
)
from app.models import Account, Hold
from app.reconcile import reconcile

app = FastAPI(
    title="Ledger",
    description="Double-entry ledger with idempotent transaction posting.",
)


def get_db():
    with SessionLocal() as session:
        yield session


class LegIn(BaseModel):
    account_id: StrictInt
    # StrictInt, so 10.5 and 10.0 are both rejected rather than quietly coerced.
    # Amounts are paise; a fractional paisa is not a thing, and silently
    # rounding one is how ledgers start losing money.
    amount_minor: StrictInt


class TransactionIn(BaseModel):
    kind: StrictStr
    legs: list[LegIn] = Field(min_length=2)


class TransactionOut(BaseModel):
    id: int
    idempotency_key: str
    kind: str


@app.post("/transactions", response_model=TransactionOut)
def create_transaction(
    body: TransactionIn,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    try:
        txn, created = post_transaction(
            db,
            idempotency_key=idempotency_key,
            kind=body.kind,
            legs=[Leg(l.account_id, l.amount_minor) for l in body.legs],
        )
    except Unbalanced as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    # 201 the first time, 200 for every replay of the same key. The body is
    # identical either way, so a retrying client needs no special handling.
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return TransactionOut(
        id=txn.id, idempotency_key=txn.idempotency_key, kind=txn.kind
    )


@app.get("/accounts/{account_id}/balance")
def get_balance(account_id: int, db: Session = Depends(get_db)):
    return {"account_id": account_id, "balance_minor": account_balance(db, account_id)}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Holds
# ---------------------------------------------------------------------------


class HoldIn(BaseModel):
    cash_account_id: StrictInt
    amount_minor: StrictInt = Field(gt=0)
    expires_at: datetime | None = None


class HoldOut(BaseModel):
    id: int
    state: str
    amount_minor: int
    captured_minor: int | None
    cash_account_id: int
    held_account_id: int


def _hold_out(hold: Hold) -> HoldOut:
    return HoldOut(
        id=hold.id,
        state=str(hold.state),
        amount_minor=hold.amount_minor,
        captured_minor=hold.captured_minor,
        cash_account_id=hold.cash_account_id,
        held_account_id=hold.held_account_id,
    )


def _require(db: Session, model, object_id: int, what: str):
    found = db.get(model, object_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{what} {object_id} not found")
    return found


@app.post("/holds", response_model=HoldOut)
def create_hold(
    body: HoldIn,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    account = _require(db, Account, body.cash_account_id, "account")
    try:
        result = place_hold(
            db,
            idempotency_key=idempotency_key,
            cash_account=account,
            amount_minor=body.amount_minor,
            expires_at=body.expires_at,
        )
    except InsufficientFunds as exc:
        # 409 rather than 422: the request is well formed, the account simply
        # cannot honour it right now, and it might succeed after a deposit.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except HoldError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    response.status_code = (
        status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    )
    return _hold_out(result.hold)


class CaptureIn(BaseModel):
    settlement_account_id: StrictInt
    #: Omit to capture the whole hold. A partial capture releases the rest.
    amount_minor: StrictInt | None = None


@app.post("/holds/{hold_id}/capture", response_model=HoldOut)
def capture(
    hold_id: int,
    body: CaptureIn,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    hold = _require(db, Hold, hold_id, "hold")
    settlement = _require(db, Account, body.settlement_account_id, "account")
    try:
        resolved, changed = capture_hold(
            db,
            hold=hold,
            settlement_account=settlement,
            idempotency_key=idempotency_key,
            amount_minor=body.amount_minor,
        )
    except HoldNotClaimable as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except HoldError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    response.status_code = status.HTTP_200_OK if not changed else status.HTTP_201_CREATED
    return _hold_out(resolved)


@app.post("/holds/{hold_id}/release", response_model=HoldOut)
def release(
    hold_id: int,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    hold = _require(db, Hold, hold_id, "hold")
    try:
        resolved, changed = release_hold(
            db, hold=hold, idempotency_key=idempotency_key
        )
    except HoldNotClaimable as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    response.status_code = status.HTTP_200_OK if not changed else status.HTTP_201_CREATED
    return _hold_out(resolved)


# ---------------------------------------------------------------------------
# Reversals and reconciliation
# ---------------------------------------------------------------------------


class ReversalIn(BaseModel):
    reason: StrictStr = ""


@app.post("/transactions/{transaction_id}/reverse", response_model=TransactionOut)
def reverse(
    transaction_id: int,
    body: ReversalIn,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    try:
        txn, created = reverse_transaction(
            db,
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            reason=body.reason,
        )
    except ReversalRefused as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return TransactionOut(
        id=txn.id, idempotency_key=txn.idempotency_key, kind=txn.kind
    )


@app.get("/accounts/{account_id}/available")
def get_available(account_id: int, db: Session = Depends(get_db)):
    """Spendable balance: what is in the cash account, held funds excluded.

    They are excluded by not being there. Placing a hold moves them out, so
    this is the same sum as the total balance with no special case -- which is
    the whole reason a hold is modelled as a transfer.
    """
    return {
        "account_id": account_id,
        "available_minor": available_balance(db, account_id),
    }


@app.get("/reconcile")
def run_reconcile(response: Response, db: Session = Depends(get_db)):
    """Every invariant, checked against the data rather than at write time.

    503 when it fails, so an uptime monitor pointed here reports a ledger that
    has drifted as an outage -- which is what it is.
    """
    report = reconcile(db)
    if not report.clean:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "clean": report.clean,
        "checks_run": report.checks_run,
        "findings": [
            {"check": f.check, "detail": f.detail} for f in report.findings
        ],
    }
