import pytest
import menace.roi_tracker as rt


def test_prediction_summary():
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.7)
    tracker.record_prediction(1.0, 0.9)
    tracker.record_class_prediction("up", "up")
    tracker.record_class_prediction("down", "up")
    stats = tracker.prediction_summary()
    assert stats["mae"] == pytest.approx(0.15)
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["class_counts"]["predicted"]["up"] == 1
    assert stats["class_counts"]["actual"]["up"] == 2

