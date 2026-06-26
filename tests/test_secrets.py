import pytest

from aiterate.config import settings
from aiterate.secrets import SecretStore


def test_production_secret_key_requires_explicit_key_or_auto_generate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "secret_key", None)
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "auto_generate_secret_key", False)

    with pytest.raises(RuntimeError, match="AIT_SECRET_KEY is required"):
        SecretStore(root=tmp_path)._fernet()


def test_auto_generate_secret_key_persists_for_docker_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "secret_key", None)
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "auto_generate_secret_key", True)

    first = SecretStore(root=tmp_path)._fernet()
    token = first.encrypt(b"saved-provider-key")
    key_path = tmp_path / "local-fernet.key"

    assert key_path.exists()
    assert SecretStore(root=tmp_path)._fernet().decrypt(token) == b"saved-provider-key"
