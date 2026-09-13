# Ledger

Double-entry ledger for a brokerage, with the invariants enforced in Postgres
rather than in Python.

The premise: *any rule that application code is responsible for is a rule some
future code path will forget.* So the rules that must never break are
constraints and triggers, application code is a source of good error messages,
and a reconciliation job checks the whole thing from the outside on the
assumption that the constraints were bypassed at some point — because they can
be, and this repo shows exactly how.

```bash
cp .env.example .env          # DATABASE_URL
alembic upgrade head
uvicorn app.api:app --reload
pytest
python -m app.reconcile       # exits non-zero if the books have drifted
```

## The four decisions

**Money is a signed `BIGINT` count of paise.** Never a float, never a `NUMERIC`
column doing duty as one. `0.1 + 0.2` is not `0.3` in binary floating point, and
a ledger that cannot represent its own balances exactly is not a ledger. At the
HTTP boundary amounts are `StrictInt`, so `10.5` and `10.0` are both rejected
rather than quietly coerced.

**The balance invariant is a deferred constraint trigger.** Entries carry signed
amounts and must sum to zero per transaction. The subtlety is *when* to check:
both legs are separate `INSERT`s inside one transaction, so a plain
`AFTER INSERT` trigger fires after the first leg and rejects a perfectly valid
write that is only half-applied.

```sql
CREATE CONSTRAINT TRIGGER entries_balance_check
    AFTER INSERT ON entries
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    EXECUTE FUNCTION assert_transaction_balances();
```

Deferring to `COMMIT` lets both legs land first. `test_invariants.py` asserts
that the *insert succeeds and the commit fails*, which is the only thing that
proves the constraint is genuinely deferred rather than merely present.

**Entries are append-only.** A `BEFORE UPDATE OR DELETE` trigger refuses
mutation outright. Corrections are compensating entries, so any account's
balance at any past moment stays reconstructible by replay.

**Exactly one SQLSTATE is swallowed.**

```python
except IntegrityError as exc:
    session.rollback()
    if _sqlstate(exc) == UNIQUE_VIOLATION:      # 23505
        winner = _find_by_key(session, idempotency_key)
        if winner is not None:
            return winner, False
    raise
```

`23505` means another caller won the race on the idempotency key, and returning
their transaction is correct. `23514` means the balance trigger rejected the
write at `COMMIT`, and it must propagate. Catching `IntegrityError` broadly
would turn an unbalanced write into "already posted" and hand the caller
somebody else's transaction — the class of bug that puts a ledger out by a few
paise and nobody notices for a month.

## Idempotency

One key posts exactly once, however many times the request arrives and however
many callers race. A retry after a timeout gets the original transaction back
with `200` instead of `201`; the body is identical, so a retrying client needs
no special handling. `test_idempotency.py` runs eight threads through a barrier
and asserts every caller receives the same transaction id with exactly one
reporting `created=True`.

Answering `409` here would be wrong. A retry after a timeout is the client doing
exactly what it should, and an error teaches it to stop.

## Holds

A brokerage cannot post a buy when an order is placed. It may not fill, may fill
partially, may be cancelled — but the cash has to stop being spendable
immediately, or the customer places two orders against one balance.

The obvious design is a `reserved_balance` column. That is a second source of
truth: it can disagree with the entries, and when it does there is no way to say
which is right. So a hold is modelled as what it actually is — **a transfer**,
from the customer's cash account to a holding account they own and cannot spend
from.

```
place    cash → held            the order was accepted
capture  held → settlement      it filled, wholly or partly
release  held → cash            it was cancelled, or expired
```

Available balance stays `SUM(entries)` on the cash account with no special case,
because held money is not there any more. The `holds` table stores no balance
that matters; it is a lifecycle pointer.

### Why a capture is the hard one

Fill callbacks are delivered at least once, sometimes after the customer has
already cancelled, and two can arrive simultaneously on different workers. So
the **state transition** is what is made atomic, and the entries follow it:

```sql
UPDATE holds
   SET state = 'captured', resolve_transaction_id = :txn, resolved_at = now()
 WHERE id = :id AND state = 'active'
```

The row count decides. One means this caller claimed the hold and may post the
legs; zero means somebody else already did, or it expired. Two concurrent
captures serialise on the row lock, and Postgres re-evaluates the `WHERE`
against the committed version, so the second matches no rows.

The opposite order — post the legs, then mark the hold — looks equivalent and is
not. Both callers would post, every transaction would balance perfectly on its
own, and the holding account would end up negative by exactly one fill.
`test_concurrent_captures_settle_exactly_once` runs eight threads at it.

State and resolution move in **one** statement. Splitting them would leave a
window where a crash strands a hold reading `captured` with no record of where
the money went — which `holds_resolution_consistent` refuses outright, and which
no amount of care in application code would prevent.

A partial capture releases the remainder in the same transaction. Left behind it
would be money the customer owns, cannot spend, and has nothing open against.

## Reversals

Nothing is edited or deleted; a reversal is a new transaction whose legs are the
original's negated, pointing at what it undid.

`reverses_transaction_id` is **`UNIQUE`**. Two concurrent reversal requests both
pass a check-then-insert, and only a unique index stops the second. The failure
mode is quiet: each reversal balances perfectly, so a per-transaction audit sees
nothing wrong, and the account is simply over-credited by the original amount.

Hold transactions are refused by the generic path. Reversing a placement would
return funds to spendable cash while the hold still reads `active` — the money
would be both spendable and claimed. Holds are undone through `release`, which
moves the state machine with the money.

## Reconciliation

A fair question: with the invariants already in the database, what is left to
reconcile? Three things.

**Constraints can be turned off.** `conftest.py` disables the append-only
trigger to truncate between tests, because there is no other way to clean a
table that refuses deletion. That is legitimate, and it is proof the capability
exists — a migration, a data fix, or a restore from a logical dump all write
rows the triggers never saw. A check that only runs at write time cannot notice
that it did not run.

**Some invariants span rows no constraint can see together.** "The holding
account's balance equals the sum of active holds against it" relates a `SUM` over
`entries` to a `SUM` over `holds`. No `CHECK` expresses it and no trigger
enforces it without serialising every write. It is the only check that compares
two independent representations of the same fact, so it is the one that would
actually catch a bug in `holds.py`.

**A ledger can satisfy every constraint and still be wrong.** A cash account
overdrawn by four lakh balances perfectly. It is still a customer spending money
they do not have.

Seven checks, each returning the offending rows rather than a boolean —
"reconciliation failed" at 3am is not actionable; "account 41 is overdrawn by
400000 paise" is. `tests/test_reconcile.py` inflicts each kind of damage on
purpose, with the triggers switched off, and requires the matching check to name
it. A reconciliation job that has only ever run against correct data is not
known to work.

## Tests

56 tests, all against real Postgres. Sqlite has no deferred constraint triggers,
no `23505`/`23514` distinction and no row-level locking under concurrent
`UPDATE`, so a suite that passed against it would be testing nothing this
project claims — and would go green on exactly the changes that break
production.

The ones that carry their weight:

- `test_invariants.py::test_unbalanced_transaction_fails_at_commit_not_at_insert`
  — proves the constraint is deferred, not merely present.
- `test_idempotency.py` — eight threads through a barrier on one key.
- `test_holds.py::test_concurrent_captures_settle_exactly_once` — eight threads,
  one hold, one fill.
- `test_reversal.py::test_concurrent_reversals_reverse_once` — the check-then-
  insert that only a unique index saves.
- `test_reconcile.py` — thirteen tests that break the ledger deliberately and
  require it to be caught.

## CI and deploy

GitHub Actions runs a real `postgres:17` service container, applies the
migrations, **runs them back down to base and up again** (a downgrade that has
never been executed is not a rollback plan, it is a comment), runs the suite,
and then runs the reconciliation job against the database the tests just
finished writing to.

`Dockerfile` builds server-side, runs as a non-root user, and applies migrations
at startup — Render's free tier has no release phase, `alembic upgrade head` is
a no-op when current, and it takes an advisory lock so two instances booting
together cannot both apply the same revision. `render.yaml` also schedules
reconciliation hourly as a cron service: it exits non-zero on drift, so a failed
run surfaces without anyone having to read the output.

## What is not here

Multi-currency (accounts carry a currency column that nothing yet enforces
across a transaction's legs), authentication, rate limiting, and a stored
balance column — balances are derived by summing entries, which is correct and
will stop being fast. `account_balance` documents that trade-off; the
reconciliation job is what would make introducing a cached balance safe.
