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


def test_entropy_ceiling_threshold_default():
    tracker = rt.ROITracker(tolerance=0.01)
    tracker.update(0.0, 0.01, metrics={"synergy_shannon_entropy": 0.2})
    _, _, _, stop = tracker.update(
        0.01, 0.02, metrics={"synergy_shannon_entropy": 0.4}
    )
    assert not stop


def test_entropy_ceiling_threshold_override():
    tracker = rt.ROITracker(entropy_threshold=0.5)
    tracker.update(0.0, 0.01, metrics={"synergy_shannon_entropy": 0.2})
    _, _, _, stop = tracker.update(
        0.01, 0.02, metrics={"synergy_shannon_entropy": 0.4}
    )
    assert stop
