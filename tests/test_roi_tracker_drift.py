from menace_sandbox.roi_tracker import ROITracker


def _patch_predictor(monkeypatch, called):
    class DummyPredictor:
        def __init__(self):
            pass

        def train(self):
            called.append(1)

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.AdaptiveROIPredictor",
        DummyPredictor,
    )


def test_drift_triggers_retrain(monkeypatch):
    called = []
    _patch_predictor(monkeypatch, called)
    tracker = ROITracker(evaluation_window=3, mae_threshold=0.1)
    tracker.record_prediction(1.0, 0.0)
    tracker.record_prediction(1.0, 0.0)
    tracker.record_prediction(1.0, 0.0)
    assert tracker.drift_flags[-1] is True
    assert called


def test_no_drift_no_retrain(monkeypatch):
    called = []
    _patch_predictor(monkeypatch, called)
    tracker = ROITracker(evaluation_window=3, mae_threshold=1.0)
    tracker.record_prediction(0.1, 0.0)
    tracker.record_prediction(0.1, 0.0)
    tracker.record_prediction(0.1, 0.0)
    assert tracker.drift_flags[-1] is False
    assert not called
