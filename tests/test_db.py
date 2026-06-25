from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from aiterate.config import settings
from aiterate import db


def test_run_migrations_is_single_flight(tmp_path, monkeypatch):
    calls = []

    def fake_upgrade(config, revision):
        _ = config, revision
        time.sleep(0.05)
        calls.append("upgrade")

    monkeypatch.setattr(db, "_migrations_applied", False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'aiterate.db'}")
    monkeypatch.setattr(db.command, "upgrade", fake_upgrade)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: db.run_migrations(), range(8)))

    assert calls == ["upgrade"]
    assert db._migrations_applied is True
