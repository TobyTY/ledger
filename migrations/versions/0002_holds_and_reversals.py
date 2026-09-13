"""Holds and reversals, with both lifecycles constrained in-database.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-13

A brokerage cannot post a buy the moment an order is placed. The order may not
fill, may fill partially, or may be cancelled -- but the cash has to stop being
spendable immediately, or the customer places two orders against one balance.

The temptation is a `reserved_balance` column on the account. That is a second
source of truth: it can disagree with the entries, and when it does there is no
way to tell which one is right. So a hold is modelled as what it actually is --
a transfer of value from the customer's cash account to a holding account they
own but cannot spend from. Available balance stays `SUM(entries)` on the cash
account, with no special cases, and the double-entry invariant covers held funds
for free.

The `holds` table therefore stores no amounts that matter. It is a LIFECYCLE
POINTER: which transaction created this hold, which one resolved it, and which
state it is in. The money lives in entries, where the trigger can see it.

TWO INVARIANTS ARE PUSHED INTO THE SCHEMA, because both are the kind that
application code gets wrong once and then gets wrong quietly:

`reverses_transaction_id` is UNIQUE. A transaction can be reversed at most once.
Without this, two concurrent reversal requests both succeed and the account ends
up over-credited by exactly the original amount -- a bug that nets to zero
against nothing and is invisible in any per-transaction check.

`holds_resolution_consistent` ties the state column to the resolution columns.
An active hold has no resolving transaction and no resolved_at; a resolved hold
has both. A row saying "captured" with a NULL resolve_transaction_id is money
that left the holding account with nothing recording where it went, and the
reconciliation job would report it as a discrepancy without being able to say
what happened.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Held cash is a distinct account type, not a flag on the cash account.
    # A balance query that forgets the flag would report held money as
    # spendable; one that forgets to include this account type simply does not
    # see it, which is the safe direction to be wrong in.
    op.execute("ALTER TYPE account_type ADD VALUE IF NOT EXISTS 'user_cash_held'")

    op.add_column(
        "transactions",
        sa.Column(
            "reverses_transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.id"),
            nullable=True,
        ),
    )
    # At most one reversal per transaction, enforced where concurrency cannot
    # argue with it.
    op.create_unique_constraint(
        "uq_transactions_reverses", "transactions", ["reverses_transaction_id"]
    )

    op.create_table(
        "holds",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column(
            "cash_account_id",
            sa.BigInteger(),
            sa.ForeignKey("accounts.id"),
            nullable=False,
        ),
        sa.Column(
            "held_account_id",
            sa.BigInteger(),
            sa.ForeignKey("accounts.id"),
            nullable=False,
        ),
        # Positive paise. The signed legs live in entries; this is the amount
        # the hold was placed for, kept here so a capture can be checked
        # against it without summing entries.
        sa.Column("amount_minor", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(), nullable=False, server_default="active"),
        sa.Column(
            "place_transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.id"),
            nullable=False,
        ),
        sa.Column(
            "resolve_transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.id"),
            nullable=True,
        ),
        sa.Column("captured_minor", sa.BigInteger(), nullable=True),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("amount_minor > 0", name="holds_amount_positive"),
        sa.CheckConstraint(
            "state IN ('active', 'captured', 'released', 'expired')",
            name="holds_state_known",
        ),
        sa.CheckConstraint(
            "captured_minor IS NULL "
            "OR (captured_minor >= 0 AND captured_minor <= amount_minor)",
            name="holds_capture_within_hold",
        ),
        sa.CheckConstraint(
            """
            (state = 'active'
                 AND resolve_transaction_id IS NULL
                 AND resolved_at IS NULL)
            OR (state <> 'active'
                 AND resolve_transaction_id IS NOT NULL
                 AND resolved_at IS NOT NULL)
            """,
            name="holds_resolution_consistent",
        ),
    )
    op.create_index("ix_holds_cash_account_id", "holds", ["cash_account_id"])
    op.create_index("ix_holds_state", "holds", ["state"])


def downgrade() -> None:
    op.drop_index("ix_holds_state", table_name="holds")
    op.drop_index("ix_holds_cash_account_id", table_name="holds")
    op.drop_table("holds")
    op.drop_constraint("uq_transactions_reverses", "transactions", type_="unique")
    op.drop_column("transactions", "reverses_transaction_id")
    # The enum value is deliberately left in place. Postgres cannot remove a
    # value from an enum without rebuilding the type, and a downgrade that
    # rewrites a column used by live data is more dangerous than an unused
    # label.
