import os
import sys
import types
os.environ.setdefault("MENACE_DB_PATH", "/tmp")
os.environ.setdefault("MENACE_SHARED_DB_PATH", "/tmp")


class DummyBuilder:
    def __init__(self):
        self.terms = []

    def refresh_db_weights(self):
        pass

    def query(self, term, **kwargs):  # pragma: no cover - simple stub
        self.terms.append(term)
        return {"snippets": [term], "metadata": {"q": term}}

# Stub loguru to avoid optional dependency
loguru_mod = types.ModuleType("loguru")
class DummyLogger:
    def add(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None

loguru_mod.logger = DummyLogger()
sys.modules.setdefault("loguru", loguru_mod)

# Stub vector_service to avoid heavy dependencies
vector_service_stub = types.SimpleNamespace(
    ContextBuilder=DummyBuilder,
    FallbackResult=object,
    ErrorResult=object,
    Retriever=object,
    EmbeddingBackfill=object,
    CognitionLayer=object,
    EmbeddableDBMixin=object,
    SharedVectorService=object,
)
sys.modules.setdefault("vector_service", vector_service_stub)

# Stub db_router to prevent filesystem access
class DummyConn:
    def execute(self, *a, **k):
        return None

    def commit(self):  # pragma: no cover - stub
        return None

class DummyRouter:
    def __init__(self, *a, **k):
        self.terms = []

    def query_all(self, term):  # pragma: no cover - stub
        self.terms.append(term)
        return {}

    def get_connection(self, name):  # pragma: no cover - stub
        return DummyConn()

db_router_stub = types.SimpleNamespace(
    DBRouter=DummyRouter,
    GLOBAL_ROUTER=DummyRouter(),
    init_db_router=lambda name: DummyRouter(),
)
sys.modules.setdefault("menace.db_router", db_router_stub)
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
    builder = DummyBuilder()
    ebot = cmb.ErrorBot(edb, context_builder=builder)
    router = DummyRouter()
    bot = cmb.CommunicationMaintenanceBot(
        mdb, ebot, repo_path, db_router=router, context_builder=builder
    )
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)

    bot.apply_hotfix("fix", lambda: None)
    rows = mdb.fetch()
    assert rows and rows[0][1] == "applied"
    df = edb.discrepancies()
    assert not df.empty
    assert "fix" in router.terms
    assert "fix" in builder.terms


def test_adjust_resources(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo = cmb.Repo.init(repo_path)
    (repo_path / "file.txt").write_text("a")
    repo.git.add(A=True)
    repo.index.commit("init")

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    router2 = DummyRouter()
    builder = DummyBuilder()
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, db_router=router2, context_builder=builder
    )
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    metrics = {"a": ResourceMetrics(cpu=1.0, memory=10.0, disk=1.0, time=1.0)}
    actions = bot.adjust_resources(metrics)
    assert actions and actions[0][0] == "a"
    assert "allocation" in router2.terms
    assert "allocation" in builder.terms


def test_check_updates(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo = cmb.Repo.init(repo_path)
    (repo_path / "file.txt").write_text("a")
    repo.git.add(A=True)
    repo.index.commit("init")

    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    router3 = DummyRouter()
    builder = DummyBuilder()
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, db_router=router3, context_builder=builder
    )
    monkeypatch.setattr(bot, "notify", lambda *a, **k: None)
    monkeypatch.setattr(bot, "notify_critical", lambda *a, **k: None)
    bot.check_updates()
    rows = mdb.fetch()
    assert rows and rows[0][0] == "update_check"
    assert "update" in router3.terms
    assert "update" in builder.terms


def test_error_bot_receives_falsey_manager(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    builder = DummyBuilder()

    class FalseyManager:
        def __init__(self) -> None:
            self.bot_registry = object()
            self.data_bot = object()

        def __bool__(self) -> bool:  # pragma: no cover - explicit behaviour
            return False

    sentinel = FalseyManager()
    captured: dict[str, object] = {}

    class DummyErrorBot:
        def __init__(self) -> None:
            self.db = types.SimpleNamespace(log_discrepancy=lambda *_a, **_k: None)

        def handle_error(self, *_a, **_k):  # pragma: no cover - stub
            return None

    def fake_construct_error_bot(*, manager, **kwargs):
        captured["manager"] = manager
        return DummyErrorBot()

    monkeypatch.setattr(cmb, "_construct_error_bot", fake_construct_error_bot)

    cmb.CommunicationMaintenanceBot(
        mdb,
        repo_path=repo_path,
        db_router=DummyRouter(),
        context_builder=builder,
        manager=sentinel,
    )

    assert captured.get("manager") is sentinel


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
        context_builder=DummyBuilder(),
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
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, context_builder=DummyBuilder()
    )
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
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, context_builder=DummyBuilder()
    )
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
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, context_builder=DummyBuilder()
    )

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
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, context_builder=DummyBuilder()
    )
    monkeypatch.setattr(bot, "evaluate_status", lambda: {"messages_last_hour": 5, "error_rate": 0.2})
    bot.cluster_data = {"nodes": 3}
    report = bot.generate_maintenance_report()
    assert "messages last hour: 5" in report.lower()
    assert "nodes=3" in report


def test_monitor_communication_alert(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    mdb = cmb.MaintenanceDB(tmp_path / "m.db")
    bot = cmb.CommunicationMaintenanceBot(
        mdb, repo_path=repo_path, context_builder=DummyBuilder()
    )

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
