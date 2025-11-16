import importlib.util
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

class DummyMicrotrend:
    def __init__(self):
        self.planner = None
        self.run_calls = []

    def run_continuous(self, interval=0.0, stop_event=None):
        self.run_calls.append(interval)

class DummyCloner:
    def __init__(self, raise_error=False):
        self.raise_error = raise_error
        self.calls = []
        self.started = False

    def start(self):
        self.started = True

    def clone_top_workflows(self, limit=3):
        self.calls.append(limit)
        if self.raise_error:
            raise RuntimeError("boom")

sys.modules["menace.microtrend_service"] = types.ModuleType("microtrend_service")
sys.modules["menace.microtrend_service"].MicrotrendService = DummyMicrotrend
sys.modules["menace.workflow_cloner"] = types.ModuleType("workflow_cloner")
sys.modules["menace.workflow_cloner"].WorkflowCloner = DummyCloner

spec = importlib.util.spec_from_file_location(
    "menace.self_evaluation_service",
    ROOT / "self_evaluation_service.py",  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["menace.self_evaluation_service"] = mod
spec.loader.exec_module(mod)


def test_on_trend_clones_limit():
    cloner = DummyCloner()
    svc = mod.SelfEvaluationService(microtrend=DummyMicrotrend(), cloner=cloner)
    svc._on_trend([1, 2])
    assert cloner.calls == [2]


def test_on_trend_default_limit():
    cloner = DummyCloner()
    svc = mod.SelfEvaluationService(microtrend=DummyMicrotrend(), cloner=cloner)
    svc._on_trend(x for x in range(5))
    assert cloner.calls == [3]


def test_on_trend_logs_error(caplog):
    cloner = DummyCloner(raise_error=True)
    svc = mod.SelfEvaluationService(microtrend=DummyMicrotrend(), cloner=cloner)
    caplog.set_level("ERROR")
    svc._on_trend([1])
    assert "clone failed" in caplog.text


def test_run_continuous_wires_components():
    micro = DummyMicrotrend()
    cloner = DummyCloner()
    svc = mod.SelfEvaluationService(microtrend=micro, cloner=cloner)
    svc.run_continuous(interval=5)
    assert micro.planner.__self__ is svc
    assert cloner.started
    assert micro.run_calls == [5]
