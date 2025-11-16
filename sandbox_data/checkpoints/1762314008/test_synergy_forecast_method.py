import importlib.util
import sys
import types
from pathlib import Path
import pytest

if "menace.roi_tracker" in sys.modules:
    import menace.roi_tracker as rt
else:  # pragma: no cover - allow running file directly
    spec = importlib.util.spec_from_file_location(
        "menace.roi_tracker",
        Path(__file__).resolve().parents[1] / "roi_tracker.py",  # path-ignore
    )
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.roi_tracker = rt
    sys.modules["menace.roi_tracker"] = rt


def test_synergy_ema(monkeypatch):
    monkeypatch.setenv("SYNERGY_FORECAST_METHOD", "ema")
    monkeypatch.setattr(rt.ROITracker, "_regression", lambda self: (None, []))
    tracker = rt.ROITracker()
    vals = [1.0, 2.0, 3.0, 4.0]
    for v in vals:
        tracker.update(0.0, v, metrics={"synergy_roi": v})
    expected = rt.ROITracker._ema(vals)
    assert tracker.predict_synergy() == pytest.approx(expected)


def test_synergy_metric_ema(monkeypatch):
    monkeypatch.setenv("SYNERGY_FORECAST_METHOD", "ema")
    monkeypatch.setattr(rt.ROITracker, "_regression", lambda self: (None, []))
    tracker = rt.ROITracker()
    vals = [0.5, 1.0, 1.5, 2.0]
    for i, v in enumerate(vals):
        tracker.update(0.0, float(i), metrics={"m": float(i), "synergy_m": v})
    expected = rt.ROITracker._ema(vals)
    assert tracker.predict_synergy_metric("m") == pytest.approx(expected)
