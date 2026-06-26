from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_model_prices"
down_revision = "0002_model_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "model_prices" not in existing:
        op.create_table(
            "model_prices",
            sa.Column("id", sa.String(length=180), primary_key=True),
            sa.Column("provider", sa.String(length=80), nullable=False),
            sa.Column("model_id", sa.String(length=255), nullable=False),
            sa.Column("currency", sa.String(length=12), nullable=False),
            sa.Column("input_per_1m_tokens", sa.Float(), nullable=False),
            sa.Column("output_per_1m_tokens", sa.Float(), nullable=False),
            sa.Column("source", sa.String(length=255), nullable=False),
            sa.Column("notes", sa.Text(), nullable=False),
            sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_model_prices_provider", "model_prices", ["provider"])
        op.create_index("ix_model_prices_model_id", "model_prices", ["model_id"])
        op.create_index("ix_model_prices_enabled", "model_prices", ["enabled"])


def downgrade() -> None:
    op.drop_table("model_prices", if_exists=True)
