from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_project_settings"
down_revision = "0003_model_prices"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "project_settings" not in existing:
        op.create_table(
            "project_settings",
            sa.Column("project_name", sa.String(length=255), primary_key=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    op.drop_table("project_settings", if_exists=True)
