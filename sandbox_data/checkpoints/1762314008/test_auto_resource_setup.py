import json
import types
from pathlib import Path
import os
import sys
import logging
import pytest

# ruff: noqa: E402

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
os.environ.setdefault("VAULT_PATH", str(Path("sessions-test.db")))
sys.modules.setdefault("menace.proxy_broker", types.ModuleType("proxy_broker"))
sys.modules["menace.proxy_broker"].fetch_free_proxies = lambda c: []
import menace.auto_resource_setup as ars


@pytest.fixture(autouse=True)
def cleanup_sessions_db():
    yield
    if Path("sessions.db").exists():
        Path("sessions.db").unlink()


def test_ensure_proxies(monkeypatch, tmp_path):
    path = tmp_path / "p.json"
    async def fake_fetch(count):
        return [types.SimpleNamespace(ip="1.1.1.1", port=80)]
    monkeypatch.setattr(ars, "fetch_free_proxies", lambda c: fake_fetch(c))
    ars.ensure_proxies(path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == "1.0"
    assert data["proxies"][0]["ip"] == "1.1.1.1"


def test_ensure_accounts(monkeypatch, tmp_path):
    path = tmp_path / "a.json"
    monkeypatch.setattr(ars, "generate_username_for_topic", lambda t: f"{t}_user")
    ars.ensure_accounts(path, topics=["tech"])
    data = json.loads(path.read_text())
    assert data["schema_version"] == "1.0"
    assert data["accounts"][0]["id"].endswith("_user")


def test_ensure_proxies_read_error_logs_warning(monkeypatch, tmp_path, caplog):
    path = tmp_path / "p.json"
    path.write_text("[]")

    async def fake_fetch(count):
        return [types.SimpleNamespace(ip="2.2.2.2", port=8080)]

    monkeypatch.setattr(ars, "fetch_free_proxies", lambda c: fake_fetch(c))
    original = Path.read_text

    def bad_read(self, *a, **k):
        if self == path:
            raise OSError("nope")
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", bad_read)
    caplog.set_level(logging.ERROR)
    ars.ensure_proxies(path)
    assert "nope" in caplog.text


def test_ensure_accounts_read_error_logs_warning(monkeypatch, tmp_path, caplog):
    path = tmp_path / "a.json"
    path.write_text("{}")
    monkeypatch.setattr(ars, "generate_username_for_topic", lambda t: f"{t}_u")
    original = Path.read_text

    def bad_read(self, *a, **k):
        if self == path:
            raise OSError("boom")
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", bad_read)
    caplog.set_level(logging.ERROR)
    ars.ensure_accounts(path, topics=["x"])
    assert "boom" in caplog.text


def test_preserve_existing(monkeypatch, tmp_path):
    path = tmp_path / "p.json"
    async def fake_fetch(count):
        return [
            types.SimpleNamespace(ip="1.1.1.1", port=80),
            types.SimpleNamespace(ip="2.2.2.2", port=81),
        ]
    monkeypatch.setattr(ars, "fetch_free_proxies", lambda c: fake_fetch(c))
    ars.ensure_proxies(path, count=1)
    ars.ensure_proxies(path, count=2, preserve_existing=True)
    data = json.loads(path.read_text())
    assert len(data["proxies"]) == 2


def test_account_preserve_existing(monkeypatch, tmp_path):
    path = tmp_path / "a.json"
    monkeypatch.setattr(ars, "generate_username_for_topic", lambda t: f"{t}_u")
    ars.ensure_accounts(path, topics=["x"], platform="Yt")
    ars.ensure_accounts(path, topics=["y"], preserve_existing=True, platform="Yt")
    data = json.loads(path.read_text())
    ids = [a["id"] for a in data["accounts"]]
    assert len(ids) == 2 and len(set(ids)) == 2
