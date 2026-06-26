from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_model_catalog"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "model_catalog" not in existing:
        op.create_table(
            "model_catalog",
            sa.Column("id", sa.String(length=160), primary_key=True),
            sa.Column("provider", sa.String(length=80), nullable=False),
            sa.Column("model_id", sa.String(length=255), nullable=False),
            sa.Column("label", sa.String(length=255), nullable=False),
            sa.Column("recommended_for", sa.JSON(), nullable=False),
            sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_model_catalog_provider", "model_catalog", ["provider"])
        op.create_index("ix_model_catalog_enabled", "model_catalog", ["enabled"])


def downgrade() -> None:
    op.drop_table("model_catalog", if_exists=True)
