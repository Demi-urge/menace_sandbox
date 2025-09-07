import asyncio
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
for name in [
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.asymmetric",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.serialization",
    "yaml",
    "numpy",
    "httpx",
    "sqlalchemy",
    "sqlalchemy.engine",
]:
    sys.modules.setdefault(name, types.ModuleType(name))
if "filelock" not in sys.modules:
    fl = types.ModuleType("filelock")
    class DummyLock:
        def __init__(self, *a, **k):
            pass
        def acquire(self, *a, **k):
            pass
        def release(self):
            pass
    fl.FileLock = DummyLock
    fl.Timeout = RuntimeError
    sys.modules["filelock"] = fl
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
cov_mod = types.ModuleType("coverage")
class _Cov:
    def __init__(self, *a, **k):
        pass
    def combine(self, *a, **k):
        pass
    def xml_report(self, *a, **k):
        pass
    def report(self, *a, **k):
        return 80.0
cov_mod.Coverage = _Cov
sys.modules.setdefault("coverage", cov_mod)

if "pydantic" not in sys.modules:
    pmod = types.ModuleType("pydantic")
    class BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def dict(self):
            return self.__dict__
        @classmethod
        def parse_obj(cls, obj):
            return cls(**obj)
    class BaseSettings:
        pass
    def Field(default=None, **kw):
        return default
    def validator(*a, **k):
        def wrap(fn):
            return fn
        return wrap
    pmod.BaseModel = BaseModel
    pmod.BaseSettings = BaseSettings
    pmod.ValidationError = Exception
    pmod.Field = Field
    pmod.validator = validator
    sys.modules["pydantic"] = pmod
    dc_mod = types.ModuleType("pydantic.dataclasses")
    dc_mod.dataclass = lambda *a, **k: (lambda f: f)
    sys.modules["pydantic.dataclasses"] = dc_mod
if "pydantic_settings" not in sys.modules:
    ps_mod = types.ModuleType("pydantic_settings")
    ps_mod.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = ps_mod

from tests.test_self_test_service_async import load_self_test_service
from tests.test_self_debugger_sandbox import (
    sds,
    DummyTelem,
    DummyEngine,
    DummyTrail,
    DummyBuilder,
)
from tests.test_self_debugger_patch_flow import FlowEngine
from tests.test_self_improvement_logging import _load_engine

sts = load_self_test_service()
sie = _load_engine()


class DummyPipe:
    def run(self, model: str, energy: int = 1):
        return sie.AutomationResult(package=None, roi=sie.ROIResult(0.0, 0.0, 0.0, 0.0, 0.0))


class DummyDiag:
    def __init__(self):
        self.metrics = types.SimpleNamespace(fetch=lambda *a, **k: [])
        self.error_bot = types.SimpleNamespace(db=types.SimpleNamespace(discrepancies=lambda: []))

    def diagnose(self):
        return []


class DummyInfo:
    def set_current_model(self, *a, **k):
        pass


class DummyCapital:
    def __init__(self):
        self.idx = 0

    def energy_score(self, **k) -> float:
        return 1.0

    def profit(self) -> float:
        val = self.idx * 0.1
        self.idx += 1
        return val

    def log_evolution_event(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self):
        self.metrics_history = {n: [0.0] for n in [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]}

    def advance(self):
        for k in self.metrics_history:
            self.metrics_history[k].append(self.metrics_history[k][-1] + 0.1)


async def _run_service_once(service: sts.SelfTestService) -> None:
    async def fake_run_once(self):
        self.results = {
            "passed": 1,
            "failed": 0,
            "coverage": 100.0,
            "runtime": 0.01,
        }
    service.__class__._run_once = fake_run_once
    await service._run_once()


def test_self_improvement_integration(monkeypatch, tmp_path):
    # SelfTestService using temporary history path
    svc = sts.SelfTestService(history_path=tmp_path / "hist.json")
    monkeypatch.setattr(sts.SelfTestService, "_docker_available", lambda self: False)
    asyncio.run(_run_service_once(svc))
    assert svc.results and svc.results["passed"] == 1

    # SelfDebuggerSandbox with dummy engine
    engine = FlowEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n    pass\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (80.0, 0.0))
    monkeypatch.setattr(engine, "_current_errors", lambda: 0, raising=False)

    dbg.analyse_and_fix()
    recs = [json.loads(r) for r in trail.records]
    assert recs and recs[-1]["result"] == "success"
    assert engine.rollback_patch_calls == ["1"]

    # SelfImprovementEngine with synergy weights
    tracker = DummyTracker()
    sie.ResearchAggregatorBot = lambda *a, **k: object()
    sie.ModelAutomationPipeline = lambda *a, **k: DummyPipe()
    sie.DiagnosticManager = lambda *a, **k: DummyDiag()
    sie.ErrorBot = lambda *a, **k: object()
    sie.ErrorDB = lambda *a, **k: object()
    sie.MetricsDB = lambda *a, **k: object()
    sie.PatchHistoryDB = lambda *a, **k: object()
    sie.InfoDB = DummyInfo
    class _Policy:
        def __init__(self, *a, **k):
            pass
        def score(self, state=None):
            return 0.0
        def update(self, *a, **k):
            pass
        def save(self):
            pass
    sie.SelfImprovementPolicy = _Policy
    sie.ConfigurableSelfImprovementPolicy = _Policy
    sie.SandboxSettings = lambda: types.SimpleNamespace(
        sandbox_data_dir=str(tmp_path),
        sandbox_score_db="db",
        synergy_weight_roi=1.0,
        synergy_weight_efficiency=1.0,
        synergy_weight_resilience=1.0,
        synergy_weight_antifragility=1.0,
        roi_ema_alpha=0.1,
        synergy_weights_lr=0.1,
    )
    monkeypatch.setattr(sie.SelfImprovementEngine, "_record_state", lambda self: None)
    sie.AutomationResult = lambda package=None, roi=None: types.SimpleNamespace(package=package, roi=roi)
    sie.ROIResult = lambda *a, **k: types.SimpleNamespace(roi=0.0)
    eng = sie.SelfImprovementEngine(
        interval=0,
        pipeline=DummyPipe(),
        diagnostics=DummyDiag(),
        info_db=DummyInfo(),
        capital_bot=DummyCapital(),
        synergy_weights_path=tmp_path / "weights.json",
        synergy_weights_lr=1.0,
    )
    sie.bootstrap = lambda: 0
    eng.tracker = tracker
    eng.synergy_learner.strategy.update = lambda *a, **k: 0.0
    eng.synergy_learner.strategy.predict = lambda *a, **k: [0.0] * 7
    eng.error_bot = types.SimpleNamespace(auto_patch_recurrent_errors=lambda: None)

    first = eng.synergy_weight_roi
    eng.run_cycle()
    tracker.advance()
    eng.run_cycle()
    second = eng.synergy_weight_roi
    assert second != first

