import types
import sys
import pytest

stub = types.ModuleType("stub")
sys.modules.setdefault("psutil", stub)
sys.modules.setdefault("networkx", stub)
sys.modules.setdefault("pandas", stub)
sys.modules.setdefault("pulp", stub)
jinja_stub = types.ModuleType("jinja2")
jinja_stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_stub)
sys.modules.setdefault("yaml", stub)
sa_stub = types.ModuleType("sqlalchemy")
engine_stub = types.ModuleType("engine")
engine_stub.Engine = object
sa_stub.engine = engine_stub
sys.modules.setdefault("sqlalchemy", sa_stub)
sys.modules.setdefault("sqlalchemy.engine", engine_stub)
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("git", stub)
stub.Repo = object
matplotlib_stub = types.ModuleType("matplotlib")
plt_stub = types.ModuleType("pyplot")
matplotlib_stub.pyplot = plt_stub  # path-ignore
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", plt_stub)  # path-ignore
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_stub)
sys.modules.setdefault("prometheus_client", stub)
stub.CollectorRegistry = object
stub.Counter = object
stub.Gauge = object
sys.modules.setdefault("sklearn", stub)
sys.modules.setdefault("sklearn.feature_extraction", stub)
sys.modules.setdefault("sklearn.feature_extraction.text", stub)
stub.TfidfVectorizer = object
sys.modules.setdefault("sklearn.cluster", stub)
stub.KMeans = object
sys.modules.setdefault("sklearn.linear_model", stub)
stub.LinearRegression = object
sys.modules.setdefault("sklearn.model_selection", stub)
stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.metrics", stub)
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = object
sys.modules.setdefault("sklearn.ensemble", stub)
stub.RandomForestClassifier = object

import menace.evolution_scheduler as sched_mod
from menace.evolution_scheduler import EvolutionScheduler


class DummyOrchestrator:
    def __init__(self) -> None:
        self.triggers = types.SimpleNamespace(error_rate=0.5, energy_threshold=0.1)
        self.improvement_engine = types.SimpleNamespace(run_cycle=lambda: None)
        self.calls = 0

    def run_cycle(self) -> None:
        self.calls += 1


class DummyCapitalBot:
    def energy_score(self, **_: object) -> float:
        return 1.0


def _stop_after_first(sched: EvolutionScheduler):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_anomaly_threshold_trigger(monkeypatch):
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=30: [{"errors": 0}]),
        engagement_delta=lambda limit=50: 0.0,
    )
    monitor = types.SimpleNamespace(detect_anomalies=lambda bot: [1, 2])
    orch = DummyOrchestrator()
    sched = EvolutionScheduler(
        orch,
        data_bot,
        DummyCapitalBot(),
        monitor=monitor,
        interval=0,
        anomaly_threshold=2,
    )
    monkeypatch.setattr(
        sched_mod.DataBot, "detect_anomalies", staticmethod(lambda df, f: [])
    )
    sched.running = True
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))
    with pytest.raises(SystemExit):
        sched._loop()
    assert orch.calls == 1


def test_engagement_trend_trigger(monkeypatch):
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=30: [{"errors": 0}]),
        engagement_delta=lambda limit=50: -0.5,
    )
    orch = DummyOrchestrator()
    sched = EvolutionScheduler(
        orch,
        data_bot,
        DummyCapitalBot(),
        interval=0,
        engagement_threshold=-0.4,
        engagement_window=3,
        anomaly_threshold=10,
    )
    monkeypatch.setattr(
        sched_mod.DataBot, "detect_anomalies", staticmethod(lambda df, f: [])
    )
    sched._engagement_history = [-0.5, -0.5]
    sched.running = True
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))
    with pytest.raises(SystemExit):
        sched._loop()
    assert orch.calls == 1


def test_loop_logs_exception(monkeypatch, caplog):
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=30: []),
        engagement_delta=lambda limit=50: 0.0,
    )
    orch = DummyOrchestrator()

    def boom() -> None:
        raise RuntimeError("fail")

    orch.run_cycle = boom
    sched = EvolutionScheduler(orch, data_bot, DummyCapitalBot(), interval=0)
    monkeypatch.setattr(
        sched_mod.DataBot, "detect_anomalies", staticmethod(lambda df, f: [])
    )
    sched.running = True
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))
    caplog.set_level("ERROR")
    with pytest.raises(SystemExit):
        sched._loop()
    assert "evolution cycle failed" in caplog.text
    assert sched.failure_count == 1

