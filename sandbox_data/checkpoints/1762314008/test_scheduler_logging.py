import os
import sys
import types
import logging
import pytest

pytest.importorskip("git")

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
os.environ.setdefault("MAINTENANCE_DISCORD_WEBHOOKS", "http://dummy")
os.environ.setdefault("MENACE_DB_PATH", "/tmp")
os.environ.setdefault("MENACE_SHARED_DB_PATH", "/tmp")
setattr(sys.modules.setdefault("menace", types.SimpleNamespace()), "RAISE_ERRORS", False)

pytest.skip("scheduler logging dependencies unavailable", allow_module_level=True)

# Stub loguru logger to avoid optional dependency requirement
loguru_mod = types.ModuleType("loguru")
class DummyLogger:
    def add(self, *a, **k):
        pass

    def __getattr__(self, name):
        def stub(*a, **k):
            return None
        return stub

loguru_mod.logger = DummyLogger()
sys.modules.setdefault("loguru", loguru_mod)

# Stub heavy dependencies from learning modules
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
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

# cryptography stubs for AuditTrail dependency
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

# Stub engine modules to avoid heavy imports
le_mod = types.ModuleType("learning_engine")
le_mod.LearningEngine = object
ue_mod = types.ModuleType("unified_learning_engine")
ue_mod.UnifiedLearningEngine = object
ae_mod = types.ModuleType("action_learning_engine")
ae_mod.ActionLearningEngine = object
sys.modules.setdefault("menace.learning_engine", le_mod)
sys.modules.setdefault("menace.unified_learning_engine", ue_mod)
sys.modules.setdefault("menace.action_learning_engine", ae_mod)

# Simple context builder stub for tests
class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def query(self, *a, **k):  # pragma: no cover - stub
        return {"snippets": [], "metadata": {}}

# Stub vector_service for downstream imports
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

    def commit(self):
        return None

class DummyRouter:
    def __init__(self, *a, **k):
        pass

    def query_all(self, term):
        return {}

    def get_connection(self, name):
        return DummyConn()

db_router_stub = types.SimpleNamespace(
    DBRouter=DummyRouter,
    GLOBAL_ROUTER=DummyRouter(),
    init_db_router=lambda name: DummyRouter(),
)
sys.modules.setdefault("menace.db_router", db_router_stub)

import menace.cross_model_scheduler as cms
import menace.model_evaluation_service as mes
import menace.communication_maintenance_bot as cmb


def _run_once(sched, module, monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        sched._run()


def test_cross_model_scheduler_logs_failure(monkeypatch, caplog):
    sched = cms._SimpleScheduler()

    def boom():
        raise RuntimeError("fail")

    sched.tasks.append((0, boom, "j"))
    sched._next_runs["j"] = 0
    caplog.set_level(logging.ERROR)
    _run_once(sched, cms, monkeypatch)
    assert "job j failed" in caplog.text


def test_model_eval_scheduler_logs_failure(monkeypatch, caplog):
    sched = mes._SimpleScheduler()

    def boom():
        raise RuntimeError("fail")

    sched.tasks.append((0, boom, "j2"))
    caplog.set_level(logging.ERROR)
    _run_once(sched, mes, monkeypatch)
    assert "job j2 failed" in caplog.text


def test_comm_maintenance_scheduler_logs_failure(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cmb, "Celery", None)
    repo_path = tmp_path / "repo"
    cmb.Repo.init(repo_path)
    router = types.SimpleNamespace(query_all=lambda t: None)
    bot = cmb.CommunicationMaintenanceBot(
        cmb.MaintenanceDB(tmp_path / "m.db"),
        repo_path=repo_path,
        db_router=router,
        context_builder=DummyBuilder(),
    )
    sched = bot.app

    calls: list[str] = []

    def boom():
        calls.append("boom")
        raise RuntimeError("fail")

    def ok():
        calls.append("ok")

    sched.tasks.append((0, boom))
    sched.tasks.append((0, ok))
    caplog.set_level(logging.ERROR)
    import time as time_mod
    monkeypatch.setattr(time_mod, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        sched._run()
    assert "task failed" in caplog.text
    assert calls == ["boom", "ok"]
