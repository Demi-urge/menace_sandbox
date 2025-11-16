import menace.roi_tracker as rt


def test_entropy_plateau_requires_full_streak():
    tracker = rt.ROITracker()
    tracker.module_entropy_deltas["a.py"] = [0.005, 0.004]  # path-ignore
    assert tracker.entropy_plateau(0.01, 3) == []
    tracker.module_entropy_deltas["a.py"].append(0.003)  # path-ignore
    assert tracker.entropy_plateau(0.01, 3) == ["a.py"]  # path-ignore


def test_entropy_plateau_resets_on_high_delta():
    tracker = rt.ROITracker()
    tracker.module_entropy_deltas["b.py"] = [0.005, 0.004, 0.02, 0.003, 0.002]  # path-ignore
    assert tracker.entropy_plateau(0.01, 3) == []
    tracker.module_entropy_deltas["b.py"].append(0.001)  # path-ignore
    assert tracker.entropy_plateau(0.01, 3) == ["b.py"]  # path-ignore


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


def test_entropy_ceiling_ratio_below_threshold():
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.1, metrics={"synergy_shannon_entropy": 1.0})
    tracker.update(0.1, 0.2, metrics={"synergy_shannon_entropy": 2.0})
    assert tracker.entropy_ceiling(0.2, window=2)


def test_entropy_ceiling_ratio_above_threshold():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"synergy_shannon_entropy": 1.0})
    tracker.update(1.0, 2.0, metrics={"synergy_shannon_entropy": 2.0})
    assert not tracker.entropy_ceiling(0.2, window=2)
