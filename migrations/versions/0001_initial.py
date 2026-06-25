from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "runs" not in existing:
        op.create_table(
            "runs",
            sa.Column("id", sa.String(length=80), primary_key=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("artifact_id", sa.String(length=80), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_runs_name", "runs", ["name"])
        op.create_index("ix_runs_artifact_id", "runs", ["artifact_id"])

    if "secrets" not in existing:
        op.create_table(
            "secrets",
            sa.Column("name", sa.String(length=255), primary_key=True),
            sa.Column("integration", sa.String(length=255), nullable=False),
            sa.Column("fingerprint", sa.String(length=80), nullable=False),
            sa.Column("encrypted_value", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.String(length=80), nullable=False),
        )

    if "audit_logs" not in existing:
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.String(length=80), primary_key=True),
            sa.Column("actor_id", sa.String(length=255), nullable=False),
            sa.Column("actor_role", sa.String(length=80), nullable=False),
            sa.Column("action", sa.String(length=255), nullable=False),
            sa.Column("target_type", sa.String(length=120), nullable=False),
            sa.Column("target_id", sa.String(length=255), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_audit_logs_actor_id", "audit_logs", ["actor_id"])
        op.create_index("ix_audit_logs_actor_role", "audit_logs", ["actor_role"])
        op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
        op.create_index("ix_audit_logs_target_type", "audit_logs", ["target_type"])
        op.create_index("ix_audit_logs_target_id", "audit_logs", ["target_id"])

    if "jobs" not in existing:
        op.create_table(
            "jobs",
            sa.Column("id", sa.String(length=80), primary_key=True),
            sa.Column("kind", sa.String(length=120), nullable=False),
            sa.Column("status", sa.String(length=80), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_jobs_kind", "jobs", ["kind"])
        op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    for table_name in ("jobs", "audit_logs", "secrets", "runs"):
        op.drop_table(table_name, if_exists=True)
