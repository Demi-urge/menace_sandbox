import sys
import types
import asyncio
import pytest

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
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

sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///tmp.db"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
jinja_mod = types.ModuleType("jinja2")
class DummyTemplate:
    def __init__(self, *a, **k):
        pass

    def render(self, *a, **k):
        return ""

jinja_mod.Template = DummyTemplate
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass

engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

import menace.resource_allocation_optimizer as rao


class DummyROIDB:
    def __init__(self) -> None:
        self.records = []
        self.weights = []

    class _Col(list):
        def mean(self) -> float:
            return sum(self) / len(self) if self else 0.0

    class _DF:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        @property
        def empty(self) -> bool:
            return not self.rows

        def __getitem__(self, key):
            return DummyROIDB._Col([r[key] for r in self.rows])

        def iterrows(self):
            for i, row in enumerate(self.rows):
                yield i, row

    def add(self, rec):
        self.records.append(rec)
        return len(self.records)

    def add_action_roi(self, *a, **k):
        pass

    def history(self, bot=None, limit=50):
        recs = [r.__dict__ for r in self.records if bot is None or r.bot == bot]
        return DummyROIDB._DF(recs[-limit:])

    def add_weight(self, bot, weight, ts=None):
        self.weights.append({"bot": bot, "weight": weight, "ts": ts})
        return len(self.weights)

    def weight_history(self, limit=50):
        return DummyROIDB._DF(self.weights[-limit:])


def test_record_and_priorities():
    db = DummyROIDB()
    fs = types.SimpleNamespace(failure_score=lambda b: 0.0)
    opt = rao.ResourceAllocationOptimizer(
        db,
        rl_model=rao.ContextualRL(),
        discrepancy_db=None,
        failure_system=fs,
        error_db=None,
    )
    opt.record_run(rao.KPIRecord("a", 10.0, 1.0, 1.0, 1.0))
    opt.record_run(rao.KPIRecord("b", 1.0, 1.0, 1.0, 1.0))
    w = opt.update_priorities(["a", "b"])
    assert w["a"] > w["b"]
    hist = db.weight_history()
    assert not hist.empty


def test_learning_convergence():
    db = DummyROIDB()
    fs = types.SimpleNamespace(failure_score=lambda b: 0.0)
    opt = rao.ResourceAllocationOptimizer(
        db,
        rl_model=rao.ContextualRL(),
        discrepancy_db=None,
        failure_system=fs,
        error_db=None,
    )
    for rec in [
        rao.KPIRecord("a", 5.0, 1.0, 1.0, 1.0),
        rao.KPIRecord("a", 8.0, 1.0, 1.0, 1.0),
        rao.KPIRecord("b", 1.0, 1.0, 1.0, 1.0),
        rao.KPIRecord("b", 0.5, 1.0, 1.0, 1.0),
    ]:
        opt.record_run(rec)
    for _ in range(5):
        w = opt.update_priorities(["a", "b"])
    assert w["a"] > w["b"]


class Resp:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "ok"}


def test_autoscale_retry(monkeypatch):
    attempts = []

    def post(url, json=None, timeout=5):
        attempts.append(True)
        if len(attempts) < 3:
            raise RuntimeError("fail")
        return Resp()

    monkeypatch.setattr(rao, "requests", types.SimpleNamespace(post=post))
    monkeypatch.setenv("AUTOSCALER_ENDPOINT", "http://api")
    opt = rao.ResourceAllocationOptimizer(DummyROIDB())
    assert opt.scale_up("m1")
    assert len(attempts) == 3


def test_autoscale_async(monkeypatch):
    calls = []

    def post(url, json=None, timeout=5):
        calls.append(True)
        return Resp()

    monkeypatch.setattr(rao, "requests", types.SimpleNamespace(post=post))
    monkeypatch.setenv("AUTOSCALER_ENDPOINT", "http://api")
    opt = rao.ResourceAllocationOptimizer(DummyROIDB())

    async def runner():
        assert await opt.scale_down_async("m2")

    asyncio.run(runner())
    assert len(calls) == 1


class FailingROI(DummyROIDB):
    def add_action_roi(self, *a, **k):
        raise RuntimeError("db fail")


def test_record_run_logs_error(caplog):
    db = FailingROI()
    caplog.set_level("ERROR")
    opt = rao.ResourceAllocationOptimizer(db)
    opt.record_run(rao.KPIRecord("x", 1.0, 1.0, 1.0, 1.0))
    assert "failed to add action ROI" in caplog.text


def test_prune_logs_and_continues(monkeypatch, caplog):
    db = DummyROIDB()
    event_bus = types.SimpleNamespace(
        publish=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        subscribe=lambda *a, **k: None,
    )
    opt = rao.ResourceAllocationOptimizer(db, event_bus=event_bus, grace_runs=0)
    monkeypatch.setattr(rao, "workflow_roi_stats", lambda *a, **k: {"roi": 0.0})
    caplog.set_level("ERROR")
    removed = opt.prune_workflows(["wf"], metrics_db=types.SimpleNamespace())
    assert removed == ["wf"]
    assert "failed to publish workflow:disabled" in caplog.text
