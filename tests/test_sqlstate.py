"""SQLSTATE extraction, against every driver shape that can produce it.

This file exists because the function it tests fails SILENTLY. `_sqlstate`
returning None does not raise anything anywhere -- it just makes
`is_unique_violation` answer False to every question, which turns the
idempotency race from "return the winner's transaction" into "report a
failure". Nothing in the happy path notices, and no exception is ever logged.

The shapes below are not invented. Each was taken from the driver in question
raising a real constraint violation against a real Postgres.
"""

from __future__ import annotations

import pytest

from app.ledger import CHECK_VIOLATION, UNIQUE_VIOLATION, _sqlstate, is_unique_violation


class FakeError(Exception):
    """Stands in for the driver's own exception type."""


def wrapped(orig):
    """SQLAlchemy wraps the driver error and exposes it as `.orig`."""
    exc = Exception("wrapped by SQLAlchemy")
    exc.orig = orig
    return exc


def psycopg3_error(code: str):
    err = FakeError("duplicate key value violates unique constraint")
    err.sqlstate = code
    return wrapped(err)


def psycopg2_error(code: str):
    err = FakeError("duplicate key value violates unique constraint")
    err.pgcode = code
    return wrapped(err)


def pg8000_error(code: str):
    # pg8000 hands back the Postgres wire-protocol error fields as a dict in
    # args[0], keyed by their single-letter identifiers. "C" is the SQLSTATE.
    return wrapped(
        FakeError(
            {
                "S": "ERROR",
                "V": "ERROR",
                "C": code,
                "M": "duplicate key value violates unique constraint",
                "F": "nbtinsert.c",
            }
        )
    )


@pytest.mark.parametrize(
    "build", [psycopg3_error, psycopg2_error, pg8000_error], ids=["psycopg3", "psycopg2", "pg8000"]
)
def test_every_driver_shape_yields_the_unique_violation_code(build):
    assert _sqlstate(build(UNIQUE_VIOLATION)) == UNIQUE_VIOLATION


@pytest.mark.parametrize(
    "build", [psycopg3_error, psycopg2_error, pg8000_error], ids=["psycopg3", "psycopg2", "pg8000"]
)
def test_every_driver_shape_yields_the_check_violation_code(build):
    assert _sqlstate(build(CHECK_VIOLATION)) == CHECK_VIOLATION


@pytest.mark.parametrize(
    "build", [psycopg3_error, psycopg2_error, pg8000_error], ids=["psycopg3", "psycopg2", "pg8000"]
)
def test_a_check_violation_is_never_mistaken_for_a_unique_one(build):
    """The distinction the whole module turns on. A unique violation means
    somebody else got there first and is recoverable; a check violation means
    the write was wrong and must propagate."""
    assert is_unique_violation(build(CHECK_VIOLATION)) is False
    assert is_unique_violation(build(UNIQUE_VIOLATION)) is True


def test_the_psycopg_only_version_would_have_returned_none_for_pg8000():
    """Pins the regression that prompted this file.

    The original implementation was `getattr(exc.orig, "sqlstate", None)`. Under
    pg8000 that is None, and None != "23505", so the idempotency race stops
    being recognised -- without raising anything.
    """
    orig = pg8000_error(UNIQUE_VIOLATION).orig
    assert getattr(orig, "sqlstate", None) is None   # what the old code read
    assert _sqlstate(pg8000_error(UNIQUE_VIOLATION)) == UNIQUE_VIOLATION  # what it reads now


def test_an_exception_with_no_driver_error_is_none_not_a_crash():
    assert _sqlstate(Exception("no orig attribute")) is None


def test_an_unrecognised_orig_shape_is_none_rather_than_a_guess():
    """Returning None is the safe answer: callers treat an unknown code as
    'not a unique violation' and let the error propagate, which is the
    conservative direction."""
    assert _sqlstate(wrapped(FakeError("just a string, no code anywhere"))) is None


def test_a_dict_without_the_C_field_is_none():
    assert _sqlstate(wrapped(FakeError({"S": "ERROR", "M": "something"}))) is None
