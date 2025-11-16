import numpy as np
import pytest
import menace.roi_tracker as rt


def test_diminishing_stable_history():
    tracker = rt.ROITracker(window=3, tolerance=0.05)
    tracker.roi_history = [0.05, 0.05, 0.05]
    assert tracker.diminishing() == pytest.approx(0.05)


def test_diminishing_noisy_history():
    tracker = rt.ROITracker(window=3, tolerance=0.05)
    tracker.roi_history = [0.2, -0.1, 0.15]
    expected = 0.05 * (1 + np.std(tracker.roi_history[-3:]))
    assert tracker.diminishing() == pytest.approx(expected)
    assert tracker.diminishing() > 0.05
