"""Initial ledger schema with the double-entry invariant enforced in-database.

Revision ID: 0001
Revises:
Create Date: 2026-09-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    account_type = postgresql.ENUM(
        "user_cash",
        "user_securities",
        "house_fees",
        "house_settlement",
        name="account_type",
    )
    account_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "accounts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "type",
            postgresql.ENUM(name="account_type", create_type=False),
            nullable=False,
        ),
        sa.Column("owner_id", sa.String(), nullable=True),
        sa.Column("currency", sa.CHAR(3), nullable=False, server_default="INR"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "entries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.id"),
            nullable=False,
        ),
        sa.Column(
            "account_id", sa.BigInteger(), sa.ForeignKey("accounts.id"), nullable=False
        ),
        # Signed minor units (paise). Debit positive, credit negative.
        sa.Column("amount_minor", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_entries_account_id", "entries", ["account_id"])
    op.create_index("ix_entries_transaction_id", "entries", ["transaction_id"])

    # ------------------------------------------------------------------
    # The double-entry invariant.
    #
    # A transaction's entries must sum to zero. The subtlety is *when* to
    # check: both legs are inserted as separate statements inside one
    # transaction, so a plain AFTER INSERT trigger fires after the first leg
    # and rejects a perfectly valid write that is only half-applied.
    #
    # DEFERRABLE INITIALLY DEFERRED moves the check to COMMIT time, once every
    # leg has landed. That is the whole trick, and it is why this constraint
    # can live in the database instead of in application code where a future
    # bug could bypass it.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION assert_transaction_balances() RETURNS TRIGGER AS $$
        DECLARE
            total BIGINT;
        BEGIN
            SELECT COALESCE(SUM(amount_minor), 0) INTO total
              FROM entries
             WHERE transaction_id = NEW.transaction_id;

            IF total <> 0 THEN
                RAISE EXCEPTION
                    'transaction % does not balance: entries sum to %',
                    NEW.transaction_id, total
                    USING ERRCODE = 'check_violation';
            END IF;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER entries_balance_check
            AFTER INSERT ON entries
            DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW
            EXECUTE FUNCTION assert_transaction_balances();
        """
    )

    # ------------------------------------------------------------------
    # Entries are append-only. Corrections are made by posting a compensating
    # entry, never by editing history -- so any account's balance at any past
    # moment stays reconstructible by replay.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION forbid_entry_mutation() RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION
                'entries are append-only; post a compensating entry instead'
                USING ERRCODE = 'raise_exception';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER entries_no_mutation
            BEFORE UPDATE OR DELETE ON entries
            FOR EACH ROW
            EXECUTE FUNCTION forbid_entry_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS entries_no_mutation ON entries")
    op.execute("DROP FUNCTION IF EXISTS forbid_entry_mutation()")
    op.execute("DROP TRIGGER IF EXISTS entries_balance_check ON entries")
    op.execute("DROP FUNCTION IF EXISTS assert_transaction_balances()")
    op.drop_index("ix_entries_transaction_id", table_name="entries")
    op.drop_index("ix_entries_account_id", table_name="entries")
    op.drop_table("entries")
    op.drop_table("transactions")
    op.drop_table("accounts")
    op.execute("DROP TYPE IF EXISTS account_type")
