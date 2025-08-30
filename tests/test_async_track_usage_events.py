import importlib.util
import sys
import types
import time
from pathlib import Path


def test_async_track_usage_emits(monkeypatch):
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []  # ensure submodules cannot be resolved
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    events: list[tuple[str, str, float]] = []

    def track_usage(module: str, impact: float) -> None:
        events.append(("track", module, impact))

    def record_output_impact(module: str, impact: float) -> None:
        events.append(("impact", module, impact))

    rr = types.ModuleType("relevancy_radar")
    rr.track_usage = track_usage
    rr.record_output_impact = record_output_impact
    monkeypatch.setitem(sys.modules, "relevancy_radar", rr)

    path = Path(__file__).resolve().parent.parent / "sandbox_runner" / "meta_logger.py"
    spec = importlib.util.spec_from_file_location("sandbox_runner.meta_logger", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.meta_logger", mod)
    spec.loader.exec_module(mod)

    mod._async_track_usage("foo", 1.0)

    for _ in range(100):  # wait for background thread
        if len(events) >= 2:
            break
        time.sleep(0.01)

    assert ("track", "foo", 1.0) in events
    assert ("impact", "foo", 1.0) in events
