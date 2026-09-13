"""The HTTP surface.

Thin tests: the money logic is covered where it lives. What is checked here is
the translation layer, which has its own ways of being wrong -- a status code
that makes a retry look like a failure, a float quietly accepted as paise, a
reconciliation endpoint that returns 200 while reporting that the books do not
balance.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_db
from app.models import Account, AccountType


@pytest.fixture
def client(session):
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def accounts(session):
    cash = Account(type=AccountType.user_cash, owner_id="api-user", currency="INR")
    house = Account(type=AccountType.house_settlement, currency="INR")
    session.add_all([cash, house])
    session.commit()
    return cash, house


def key() -> str:
    return f"api-{uuid.uuid4()}"


def deposit(client, cash, house, paise=10_000_00):
    return client.post(
        "/transactions",
        headers={"Idempotency-Key": key()},
        json={
            "kind": "deposit",
            "legs": [
                {"account_id": cash.id, "amount_minor": paise},
                {"account_id": house.id, "amount_minor": -paise},
            ],
        },
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_a_deposit_lands(client, accounts):
    cash, house = accounts
    assert deposit(client, cash, house).status_code == 201
    body = client.get(f"/accounts/{cash.id}/balance").json()
    assert body["balance_minor"] == 10_000_00


def test_a_replayed_key_answers_200_not_409(client, accounts):
    """A retry after a timeout is not an error. It is the client doing exactly
    what it should, and answering 409 would teach it to stop."""
    cash, house = accounts
    idem = key()
    payload = {
        "kind": "deposit",
        "legs": [
            {"account_id": cash.id, "amount_minor": 500_00},
            {"account_id": house.id, "amount_minor": -500_00},
        ],
    }
    first = client.post("/transactions", headers={"Idempotency-Key": idem}, json=payload)
    second = client.post("/transactions", headers={"Idempotency-Key": idem}, json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json() == second.json()
    assert client.get(f"/accounts/{cash.id}/balance").json()["balance_minor"] == 500_00


def test_a_fractional_paisa_is_refused(client, accounts):
    """StrictInt at the boundary. 10.5 paise does not exist, and rounding one
    silently is how a ledger starts losing money."""
    cash, house = accounts
    response = client.post(
        "/transactions",
        headers={"Idempotency-Key": key()},
        json={
            "kind": "deposit",
            "legs": [
                {"account_id": cash.id, "amount_minor": 10.5},
                {"account_id": house.id, "amount_minor": -10.5},
            ],
        },
    )
    assert response.status_code == 422


def test_an_unbalanced_request_is_refused(client, accounts):
    cash, house = accounts
    response = client.post(
        "/transactions",
        headers={"Idempotency-Key": key()},
        json={
            "kind": "deposit",
            "legs": [
                {"account_id": cash.id, "amount_minor": 100},
                {"account_id": house.id, "amount_minor": -99},
            ],
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Holds
# ---------------------------------------------------------------------------


def test_the_hold_lifecycle_over_http(client, accounts):
    cash, house = accounts
    deposit(client, cash, house)

    placed = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": cash.id, "amount_minor": 3_000_00},
    )
    assert placed.status_code == 201
    hold_id = placed.json()["id"]

    available = client.get(f"/accounts/{cash.id}/available").json()
    assert available["available_minor"] == 7_000_00

    captured = client.post(
        f"/holds/{hold_id}/capture",
        headers={"Idempotency-Key": key()},
        json={"settlement_account_id": house.id, "amount_minor": 1_200_00},
    )
    assert captured.status_code == 201
    assert captured.json()["state"] == "captured"
    assert captured.json()["captured_minor"] == 1_200_00

    # The unfilled 1,800 came straight back.
    assert (
        client.get(f"/accounts/{cash.id}/available").json()["available_minor"]
        == 8_800_00
    )


def test_a_hold_beyond_the_balance_is_409(client, accounts):
    """Not 422. The request is well formed; the account cannot honour it right
    now, and it might succeed after a deposit."""
    cash, house = accounts
    deposit(client, cash, house, 100_00)
    response = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": cash.id, "amount_minor": 500_00},
    )
    assert response.status_code == 409


def test_capturing_a_released_hold_is_409(client, accounts):
    cash, house = accounts
    deposit(client, cash, house)
    hold_id = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": cash.id, "amount_minor": 1_000_00},
    ).json()["id"]

    client.post(f"/holds/{hold_id}/release", headers={"Idempotency-Key": key()})
    response = client.post(
        f"/holds/{hold_id}/capture",
        headers={"Idempotency-Key": key()},
        json={"settlement_account_id": house.id},
    )
    assert response.status_code == 409


def test_a_hold_on_a_missing_account_is_404(client):
    response = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": 9_999_999, "amount_minor": 100},
    )
    assert response.status_code == 404


def test_a_non_positive_hold_is_refused_by_the_schema(client, accounts):
    cash, _ = accounts
    response = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": cash.id, "amount_minor": 0},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Reversal and reconciliation
# ---------------------------------------------------------------------------


def test_reversing_over_http(client, accounts):
    cash, house = accounts
    txn_id = deposit(client, cash, house).json()["id"]

    reversed_once = client.post(
        f"/transactions/{txn_id}/reverse",
        headers={"Idempotency-Key": key()},
        json={"reason": "duplicate deposit"},
    )
    assert reversed_once.status_code == 201
    assert client.get(f"/accounts/{cash.id}/balance").json()["balance_minor"] == 0

    again = client.post(
        f"/transactions/{txn_id}/reverse",
        headers={"Idempotency-Key": key()},
        json={"reason": "duplicate deposit"},
    )
    assert again.status_code == 200
    assert again.json()["id"] == reversed_once.json()["id"]
    assert client.get(f"/accounts/{cash.id}/balance").json()["balance_minor"] == 0


def test_reversing_a_hold_transaction_is_409(client, accounts, session):
    cash, house = accounts
    deposit(client, cash, house)
    hold_id = client.post(
        "/holds",
        headers={"Idempotency-Key": key()},
        json={"cash_account_id": cash.id, "amount_minor": 1_000_00},
    ).json()["id"]

    from app.models import Hold

    place_txn_id = session.get(Hold, hold_id).place_transaction_id
    response = client.post(
        f"/transactions/{place_txn_id}/reverse",
        headers={"Idempotency-Key": key()},
        json={},
    )
    assert response.status_code == 409


def test_reconcile_reports_clean(client, accounts):
    cash, house = accounts
    deposit(client, cash, house)
    response = client.get("/reconcile")

    assert response.status_code == 200
    assert response.json()["clean"] is True
    assert len(response.json()["checks_run"]) == 7


def test_reconcile_answers_503_when_the_books_have_drifted(
    client, accounts, session
):
    """So an uptime monitor pointed at this treats a drifted ledger as an
    outage, which is what it is."""
    from sqlalchemy import text

    cash, house = accounts
    deposit(client, cash, house)

    for trigger in ("entries_balance_check", "entries_no_mutation"):
        session.execute(text(f"ALTER TABLE entries DISABLE TRIGGER {trigger}"))
    session.execute(
        text("INSERT INTO transactions (idempotency_key, kind) VALUES ('x', 'bad')")
    )
    txn_id = session.execute(
        text("SELECT id FROM transactions WHERE idempotency_key = 'x'")
    ).scalar_one()
    session.execute(
        text(
            "INSERT INTO entries (transaction_id, account_id, amount_minor) "
            "VALUES (:t, :a, 1)"
        ),
        {"t": txn_id, "a": cash.id},
    )
    for trigger in ("entries_balance_check", "entries_no_mutation"):
        session.execute(text(f"ALTER TABLE entries ENABLE TRIGGER {trigger}"))
    session.commit()
    session.expire_all()

    response = client.get("/reconcile")
    assert response.status_code == 503
    assert response.json()["clean"] is False
    assert response.json()["findings"]
