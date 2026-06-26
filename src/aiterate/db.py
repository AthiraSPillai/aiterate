from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Iterator

from alembic import command
from alembic.config import Config
from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from aiterate.config import settings


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    artifact_id: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class SecretRecord(Base):
    __tablename__ = "secrets"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    integration: Mapped[str] = mapped_column(String(255))
    fingerprint: Mapped[str] = mapped_column(String(80))
    encrypted_value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(String(80))


class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(255), index=True)
    actor_role: Mapped[str] = mapped_column(String(80), index=True)
    action: Mapped[str] = mapped_column(String(255), index=True)
    target_type: Mapped[str] = mapped_column(String(120), index=True)
    target_id: Mapped[str] = mapped_column(String(255), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    kind: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ModelCatalogRecord(Base):
    __tablename__ = "model_catalog"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), index=True)
    model_id: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(255))
    recommended_for: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[int] = mapped_column(Integer, default=1, index=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ModelPriceRecord(Base):
    __tablename__ = "model_prices"

    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), index=True)
    model_id: Mapped[str] = mapped_column(String(255), index=True)
    currency: Mapped[str] = mapped_column(String(12), default="USD")
    input_per_1m_tokens: Mapped[float] = mapped_column()
    output_per_1m_tokens: Mapped[float] = mapped_column()
    source: Mapped[str] = mapped_column(String(255), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[int] = mapped_column(Integer, default=1, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ProjectSettingsRecord(Base):
    __tablename__ = "project_settings"

    project_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)
_migrations_applied = False
_migration_lock = Lock()


def init_db() -> None:
    run_migrations()


def run_migrations() -> None:
    global _migrations_applied
    if _migrations_applied:
        return
    with _migration_lock:
        if _migrations_applied:
            return
        _ensure_database_parent(settings.database_url)
        migration_dir = Path(__file__).parent / "migrations"
        config = Config()
        config.set_main_option("sqlalchemy.url", settings.database_url)
        config.set_main_option("script_location", str(migration_dir))
        command.upgrade(config, "head")
        _migrations_applied = True


def _ensure_database_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    database = make_url(database_url).database
    if not database or database == ":memory:":
        return
    db_path = Path(database)
    if db_path.name and db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
