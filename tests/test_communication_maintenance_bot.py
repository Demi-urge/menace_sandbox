import os
import sys
import types


class DummyBuilder:
    def refresh_db_weights(self):
        pass
import logging
from datetime import datetime
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
os.environ.setdefault("MAINTENANCE_DISCORD_WEBHOOKS", "http://dummy")
pytest.importorskip("git")

sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules["networkx"].DiGraph = object
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
pandas_mod = types.ModuleType("pandas")
class DummyDF:
    def __init__(self, *a, **k):
        pass
    @property
    def empty(self):
        return False
pandas_mod.DataFrame = DummyDF
pandas_mod.read_sql = lambda *a, **k: DummyDF()
pandas_mod.read_csv = lambda *a, **k: DummyDF()
sys.modules.setdefault("pandas", pandas_mod)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngineMod:
    pass
engine_mod.Engine = DummyEngineMod
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

import menace.communication_maintenance_bot as cmb
from menace.communication_maintenance_bot import PING_FAILURE_LIMIT

from dataclasses import dataclass

@dataclass
class ResourceMetrics:
    cpu: float
    memory: float
    disk: float
    time: float


def test_hotfix_logs(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo = cmb.Repo.init(repo_path)
    (repo_path / "file.txt").write_text("a")
    repo.git.add(A=True)
    repo.index.commit("init")

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    edb = cmb.ErrorDB(tmp_path / "e.db")
    ebot = cmb.ErrorBot(edb, context_builder=DummyBuilder())
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = cmb.CommunicationMaintenanceBot(mdb, ebot, repo_path, db_router=router)
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)

    bot.apply_hotfix("fix", lambda: None)
    rows = mdb.fetch()
    assert rows and rows[0][1] == "applied"
    df = edb.discrepancies()
    assert not df.empty
    assert "fix" in router.terms


def test_adjust_resources(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo = cmb.Repo.init(repo_path)
    (repo_path / "file.txt").write_text("a")
    repo.git.add(A=True)
    repo.index.commit("init")

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    router2 = types.SimpleNamespace(terms=[])
    router2.query_all = lambda term: router2.terms.append(term) or {}
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path, db_router=router2)
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    metrics = {"a": ResourceMetrics(cpu=1.0, memory=10.0, disk=1.0, time=1.0)}
    actions = bot.adjust_resources(metrics)
    assert actions and actions[0][0] == "a"
    assert "allocation" in router2.terms


def test_check_updates(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo = cmb.Repo.init(repo_path)
    (repo_path / "file.txt").write_text("a")
    repo.git.add(A=True)
    repo.index.commit("init")

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    router3 = types.SimpleNamespace(terms=[])
    router3.query_all = lambda term: router3.terms.append(term) or {}
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path, db_router=router3)
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    bot.check_updates()
    rows = mdb.fetch()
    assert rows and rows[0][0] == "update_check"
    assert "update" in router3.terms


def test_subscribe_failures_logged(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cmb, "Celery", None)
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)

    class BadBus:
        def subscribe(self, *a, **k):
            raise RuntimeError("fail")

    class BadMem:
        def subscribe(self, *a, **k):
            raise RuntimeError("fail")

    caplog.set_level(logging.ERROR)
    cmb.CommunicationMaintenanceBot(
        cmb.MaintenanceDB(tmp_path / "c.db"),
        repo_path=repo_path,
        event_bus=BadBus(),
        memory_mgr=BadMem(),
    )
    assert "event bus subscription failed" in caplog.text
    assert "memory subscription failed" in caplog.text


def test_ping_bots(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)

    calls = []

    def fake_get(url, timeout=5):
        calls.append(url)
        return types.SimpleNamespace(status_code=200, json=lambda: {"pong": True}, text="ok")

    monkeypatch.setattr(cmb.requests, "get", fake_get)
    monkeypatch.setattr(cmb.time, "sleep", lambda s: None)

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path)
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    bot.bot_urls = {"b1": "http://b1"}
    results = bot.ping_bots()

    assert results == {"b1": True}
    assert calls == ["http://b1"]


def test_ping_bots_failure_removal(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)

    def fail_get(url, timeout=5):
        raise Exception("boom")

    monkeypatch.setattr(cmb.requests, "get", fail_get)
    monkeypatch.setattr(cmb.time, "sleep", lambda s: None)

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path)
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    bot.bot_urls = {"b1": "http://b1"}
    bot.fail_counts = {"b1": PING_FAILURE_LIMIT - 1}
    bot.ping_bots()

    assert "b1" not in bot.bot_urls


def test_evaluate_status(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path)

    now = datetime.utcnow().isoformat()
    data = [
        {"timestamp": now, "message": "ok"},
        {"timestamp": now, "message": "error: boom"},
    ]
    monkeypatch.setattr(bot.comm_store, "load", lambda: data)
    metrics = bot.evaluate_status()
    assert metrics["messages_last_hour"] == 2
    assert metrics["error_rate"] == 0.5


def test_generate_maintenance_report(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path)
    monkeypatch.setattr(bot, "evaluate_status", lambda: {"messages_last_hour": 5, "error_rate": 0.2})
    bot.cluster_data = {"nodes": 3}
    report = bot.generate_maintenance_report()
    assert "messages last hour: 5" in report.lower()
    assert "nodes=3" in report


def test_monitor_communication_alert(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(mdb, repo_path=repo_path)

    monkeypatch.setattr(bot, "evaluate_status", lambda: {"messages_last_hour": 10, "error_rate": 0.3})
    called = {}

    def esc(msg, severity=cmb.Severity.CRITICAL):
        called["msg"] = msg
        called["severity"] = severity

    monkeypatch.setattr(bot, "escalate_error", esc)
    bot.monitor_communication()
    assert "high communication error rate" in called.get("msg", "").lower()


def test_maintenance_db_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAINTENANCE_DB", raising=False)
    mdb = cmb.MaintenanceDB()
    assert mdb.path.resolve() == (tmp_path / "maintenance.db").resolve()
