"""Database engine and session factory.

PSYCOPG IS THE DRIVER. pg8000 is a fallback, and only a fallback.

psycopg3 is what CI runs, what the Render blueprint runs, and what every number
in the README was measured against. It is a C extension over libpq and it is
faster than the alternative by enough to matter on a suite that opens thousands
of transactions.

The fallback exists because of a real failure on one machine. Windows
Application Control deleted the libpq DLLs that `psycopg[binary]` ships,
immediately after pip wrote them: `pip install` reported success, the wheel's
RECORD listed three .dll files, and the installed directory contained none of
them. Every import then failed with

    ImportError: no pq wrapper available.
    - couldn't import psycopg 'binary' implementation: DLL load failed while
      importing pq: An Application Control policy has blocked this file.

That is a machine policy rather than a bug here, and it cannot be fixed from
Python. pg8000 is pure Python, needs no DLL, and so is not blocked.

SELECTING IT IS DELIBERATE AND LOUD. Either `LEDGER_DRIVER=pg8000` in the
environment, or automatically when psycopg genuinely cannot be imported -- and
that case prints a warning, because a silent driver substitution is how someone
spends an hour debugging a behaviour difference they were never told about.

THE TWO DRIVERS ARE NOT INTERCHANGEABLE, which is why `app.ledger._sqlstate`
reads the error code three different ways. pg8000 exposes no `.sqlstate`
attribute at all, and it raises ProgrammingError rather than IntegrityError for
a check violation. The first of those would quietly disable the idempotency
race handling -- `is_unique_violation` would answer False to every question --
so it is handled there rather than here.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

PREFERRED = "psycopg"
FALLBACK = "pg8000"


def _psycopg_importable() -> bool:
    try:
        import psycopg  # noqa: F401
    except Exception:
        # Deliberately broad. A missing package raises ImportError, a blocked
        # DLL raises ImportError with a different message, and a half-installed
        # wheel can raise almost anything. All of them mean the same thing here.
        return False
    return True


def select_driver() -> str:
    """Which SQLAlchemy dialect suffix to use."""
    forced = os.environ.get("LEDGER_DRIVER", "").strip().lower()
    if forced in (PREFERRED, FALLBACK):
        return forced
    if forced:
        raise RuntimeError(
            f"LEDGER_DRIVER={forced!r} is not recognised; use {PREFERRED!r} or {FALLBACK!r}"
        )

    if _psycopg_importable():
        return PREFERRED

    print(
        f"WARNING: psycopg could not be imported, so falling back to {FALLBACK}.\n"
        f"         That is slower and is NOT what CI runs. If the cause is an\n"
        f"         Application Control policy deleting libpq, it is a machine\n"
        f"         setting rather than anything in this repository.",
        file=sys.stderr,
    )
    return FALLBACK


def database_url(driver: str | None = None) -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and paste your "
            "Postgres connection string into it."
        )

    driver = driver or select_driver()

    # Accept the postgres:// form that hosted providers hand out and normalise it
    # to the driver we actually use.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", f"postgresql+{driver}://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", f"postgresql+{driver}://", 1)
    return url


def _pg8000_args(url: str) -> dict:
    # psycopg reads sslmode straight out of the URL. pg8000 does not understand
    # the parameter and raises on it, so it is stripped from the URL and
    # translated into the ssl_context pg8000 does understand. Neon requires TLS,
    # so dropping the parameter without replacing it would silently downgrade
    # the connection rather than fail it -- which is the worse outcome.
    return {"ssl_context": True} if "sslmode" in url else {}


DRIVER = select_driver()
_raw = database_url(DRIVER)

if DRIVER == FALLBACK:
    engine = create_engine(
        _raw.split("?")[0],
        pool_pre_ping=True,
        future=True,
        connect_args=_pg8000_args(_raw),
    )
else:
    engine = create_engine(_raw, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def get_session() -> Session:
    return SessionLocal()
