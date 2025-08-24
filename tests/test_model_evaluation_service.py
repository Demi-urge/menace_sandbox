import sys
import types

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

import menace.model_evaluation_service as mes
from menace.model_evaluation_service import ModelEvaluationService
from menace.cross_model_comparator import CrossModelComparator
from menace.unified_event_bus import UnifiedEventBus
from menace.evaluation_worker import EvaluationWorker
from menace.evaluation_history_db import EvaluationHistoryDB
import db_router
from menace.evaluation_manager import EvaluationManager


class DummyManager:
    def __init__(self) -> None:
        self.engines = {
            "A": types.SimpleNamespace(evaluate=lambda: {"cv_score": 0.1})
        }
        self.history = {"A": []}
        self.db = None
        self._best_name = None
        self._best_score = float("-inf")

    def evaluate_all(self):
        return {"A": {"cv_score": 0.1}}


class DummyHistory:
    def deployment_weights(self):
        return {"A": 0.4}


class DummyCloner:
    def __init__(self):
        self.called = False

    def clone_top_workflows(self, limit=1):
        self.called = True


class DummyRollback:
    def __init__(self):
        self.calls = []

    def auto_rollback(self, patch_id, nodes):
        self.calls.append((patch_id, tuple(nodes)))
        return True


def test_run_cycle_triggers_rollback(monkeypatch):
    rb = DummyRollback()
    comparator = CrossModelComparator(
        pathways=None,
        history=DummyHistory(),
        deployer=None,
        rollback_mgr=rb,
    )
    bus = UnifiedEventBus()
    manager = DummyManager()
    service = ModelEvaluationService(
        manager=manager,
        comparator=comparator,
        cloner=DummyCloner(),
        event_bus=bus,
    )
    EvaluationWorker(bus, manager)

    service.run_cycle()

    assert rb.calls


def test_scheduler_adds_job(monkeypatch):
    service = ModelEvaluationService(manager=DummyManager())
    monkeypatch.setattr(mes, "BackgroundScheduler", None)
    recorded = {}

    def fake_add_job(self, func, interval, id):
        recorded["func"] = func
        recorded["interval"] = interval
        recorded["id"] = id

    monkeypatch.setattr(mes._SimpleScheduler, "add_job", fake_add_job)
    called = {"n": 0}
    monkeypatch.setattr(service, "run_cycle", lambda: called.update(n=called["n"] + 1))
    service.run_continuous(interval=42)
    assert recorded["interval"] == 42
    assert recorded["id"] == "model_evaluation"
    recorded["func"]()
    assert called["n"] == 1


def test_multi_node_results_affect_weights(tmp_path):
    bus = UnifiedEventBus()
    router = db_router.DBRouter(
        "mes", str(tmp_path / "hist.db"), str(tmp_path / "hist.db")
    )
    db = EvaluationHistoryDB(router=router)
    comparator = CrossModelComparator(pathways=None, history=db, deployer=None)
    manager = DummyManager()
    manager.engines = {"A": object(), "B": object()}
    service = ModelEvaluationService(manager=manager, comparator=comparator, event_bus=bus)

    m1 = EvaluationManager()
    m1.engines = {"A": types.SimpleNamespace(evaluate=lambda: {"cv_score": 0.2})}
    EvaluationWorker(bus, m1)

    m2 = EvaluationManager()
    m2.engines = {"B": types.SimpleNamespace(evaluate=lambda: {"cv_score": 0.8})}
    EvaluationWorker(bus, m2)

    service.run_cycle()

    weights = db.deployment_weights()
    assert weights["B"] == 1.0
    assert weights["A"] < 1.0
