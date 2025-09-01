import importlib
import sys
import types
import time


def test_async_track_usage_emits(monkeypatch):
    events: list[tuple[str, str, float]] = []

    def track_usage(module: str, impact: float) -> None:
        events.append(("track", module, impact))

    def record_output_impact(module: str, impact: float) -> None:
        events.append(("impact", module, impact))

    rr = types.ModuleType("relevancy_radar")
    rr.track_usage = track_usage
    rr.record_output_impact = record_output_impact
    rr.RelevancyRadar = object  # placeholder for other imports
    rr.evaluate_final_contribution = lambda *a, **k: None
    rr.radar = None
    monkeypatch.setitem(sys.modules, "relevancy_radar", rr)
    monkeypatch.setenv("SANDBOX_ENABLE_RELEVANCY_RADAR", "1")

    arp = types.ModuleType("adaptive_roi_predictor")
    arp.load_training_data = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "adaptive_roi_predictor", arp)

    cycle = importlib.reload(importlib.import_module("sandbox_runner.cycle"))
    cycle._async_track_usage("foo", 1.0)

    for _ in range(100):  # wait for background thread
        if ("track", "foo", 1.0) in events and ("impact", "foo", 1.0) in events:
            break
        time.sleep(0.01)

    assert ("track", "foo", 1.0) in events
    assert ("impact", "foo", 1.0) in events
