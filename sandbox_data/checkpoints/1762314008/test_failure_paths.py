import types
import sys
import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _DummyErrorLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

sys.modules.setdefault("error_logger", types.SimpleNamespace(ErrorLogger=_DummyErrorLogger))
sys.modules.setdefault("adaptive_roi_predictor", types.SimpleNamespace(load_training_data=lambda: None))

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", menace_pkg)

sandbox_pkg = types.ModuleType("menace.sandbox_runner")
sandbox_pkg.__path__ = [str(ROOT / "sandbox_runner")]
sys.modules.setdefault("menace.sandbox_runner", sandbox_pkg)

environment = importlib.import_module("menace.sandbox_runner.environment")


def test_radar_worker_start_failure_propagates(monkeypatch):
    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(environment.threading, "Thread", FailingThread)
    counter = types.SimpleNamespace(count=0)
    monkeypatch.setattr(
        environment,
        "sandbox_crashes_total",
        types.SimpleNamespace(inc=lambda: setattr(counter, "count", counter.count + 1)),
    )
    worker = environment._RadarWorker()
    with pytest.raises(RuntimeError):
        worker.__enter__()
    assert counter.count == 1
