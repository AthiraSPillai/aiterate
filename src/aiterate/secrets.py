from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field
from sqlalchemy import select

from aiterate.config import settings
from aiterate.db import SecretRecord, session_scope
from aiterate.secret_adapters import build_managed_secret_adapter


class SecretInput(BaseModel):
    name: str
    value: str = Field(min_length=1)
    integration: str


class SecretMetadata(BaseModel):
    name: str
    integration: str
    fingerprint: str
    updated_at: str


class SecretStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.storage_dir / "secrets"

    def upsert(self, secret: SecretInput) -> SecretMetadata:
        external = settings.secret_provider in {"vault", "aws", "azure", "gcp"}
        if external:
            build_managed_secret_adapter().put(secret.name, secret.value)
            encrypted = "__managed_secret__"
        else:
            encrypted = self._fernet().encrypt(secret.value.encode("utf-8")).decode("utf-8")
        metadata = SecretMetadata(
            name=secret.name,
            integration=secret.integration,
            fingerprint=_fingerprint(secret.value),
            updated_at=datetime.now(UTC).isoformat(),
        )
        with session_scope() as session:
            session.merge(
                SecretRecord(
                    name=metadata.name,
                    integration=metadata.integration,
                    fingerprint=metadata.fingerprint,
                    encrypted_value=encrypted,
                    updated_at=metadata.updated_at,
                )
            )
        return metadata

    def list(self) -> list[SecretMetadata]:
        with session_scope() as session:
            records = session.scalars(select(SecretRecord).order_by(SecretRecord.integration, SecretRecord.name)).all()
        return [
            SecretMetadata(
                name=record.name,
                integration=record.integration,
                fingerprint=record.fingerprint,
                updated_at=record.updated_at,
            )
            for record in records
        ]

    def get_value(self, name: str) -> str | None:
        if settings.secret_provider in {"vault", "aws", "azure", "gcp"}:
            value = build_managed_secret_adapter().get(name)
            if value:
                return value
        with session_scope() as session:
            record = session.get(SecretRecord, name)
        if not record:
            return None
        if record.encrypted_value == "__managed_secret__":
            return None
        return self._fernet().decrypt(record.encrypted_value.encode("utf-8")).decode("utf-8")

    def delete(self, name: str) -> bool:
        deleted = False
        external = settings.secret_provider in {"vault", "aws", "azure", "gcp"}
        if external:
            try:
                build_managed_secret_adapter().delete(name)
            except Exception:
                pass
        with session_scope() as session:
            record = session.get(SecretRecord, name)
            if record:
                session.delete(record)
                deleted = True
        return deleted

    def _fernet(self) -> Fernet:
        key = settings.secret_key
        if not key:
            if settings.environment == "production":
                raise RuntimeError("AIT_SECRET_KEY is required for encrypted secret storage.")
            key_path = self.root / "local-dev.key"
            self.root.mkdir(parents=True, exist_ok=True)
            if not key_path.exists():
                key_path.write_bytes(Fernet.generate_key())
            key = key_path.read_text(encoding="utf-8")
        return Fernet(key.encode("utf-8"))


def _fingerprint(value: str) -> str:
    prefix = value[:4] if len(value) >= 4 else "****"
    suffix = value[-4:] if len(value) >= 4 else "****"
    return f"{prefix}...{suffix}"
