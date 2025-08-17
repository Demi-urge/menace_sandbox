import menace.roi_tracker as rt


def test_entropy_plateau_requires_full_streak():
    tracker = rt.ROITracker()
    tracker.module_entropy_deltas["a.py"] = [0.005, 0.004]
    assert tracker.entropy_plateau(0.01, 3) == []
    tracker.module_entropy_deltas["a.py"].append(0.003)
    assert tracker.entropy_plateau(0.01, 3) == ["a.py"]


def test_entropy_plateau_resets_on_high_delta():
    tracker = rt.ROITracker()
    tracker.module_entropy_deltas["b.py"] = [0.005, 0.004, 0.02, 0.003, 0.002]
    assert tracker.entropy_plateau(0.01, 3) == []
    tracker.module_entropy_deltas["b.py"].append(0.001)
    assert tracker.entropy_plateau(0.01, 3) == ["b.py"]
