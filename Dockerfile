# Build context is the repo root.
FROM python:3.13-slim AS base

# Bytecode written at build time rather than on every cold start, and no
# buffering so logs from a crashing container actually arrive.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Requirements first, so a code change does not reinstall the dependency tree.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

# Not root. A ledger process has no business being able to write to its own
# source tree.
RUN useradd --create-home --uid 10001 ledger && chown -R ledger:ledger /srv
USER ledger

EXPOSE 8000

# Migrations run at startup rather than in a separate release step, because
# Render's free tier has no release phase. `alembic upgrade head` is a no-op
# when the database is current, and it takes an advisory lock, so two
# instances booting together cannot both apply the same revision.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
