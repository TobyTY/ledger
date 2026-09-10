"""HTTP surface.

Idempotency is carried in the ``Idempotency-Key`` header rather than the body,
matching the convention Stripe popularised -- it is a property of the request,
not of the money being moved.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, Field, StrictInt, StrictStr
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.ledger import Leg, Unbalanced, account_balance, post_transaction

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
