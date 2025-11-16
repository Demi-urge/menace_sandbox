import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "baseline_tracker",
    Path(__file__).resolve().parents[1] / "self_improvement" / "baseline_tracker.py",  # path-ignore
)
baseline_tracker = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(baseline_tracker)  # type: ignore[attr-defined]
BaselineTracker = baseline_tracker.BaselineTracker


def test_momentum_tracking():
    bt = BaselineTracker(window=5)
    for roi in [1.0, 1.1, 0.9, 1.0]:
        bt.update(roi=roi)
    assert bt.success_count == 2
    assert bt.cycle_count == 4
    assert bt.momentum == pytest.approx(2 / 5)
    hist = bt.to_dict()["momentum"]
    assert len(hist) == 4
    assert hist[-1] == bt.momentum
    assert bt.get("momentum") == sum(hist) / len(hist)
