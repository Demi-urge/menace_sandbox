import pytest
from pathlib import Path
import dynamic_path_router
import sys

sys.modules["menace"].RAISE_ERRORS = False

dynamic_path_router.resolve_path = lambda p: Path(p).resolve()
dynamic_path_router.path_for_prompt = lambda p: Path(p).resolve().as_posix()

for p in [
    Path("workflow_roi_history.json"),
    Path("menace_roi_tracker_local.db"),
    Path("shared/global.db"),
]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()

import menace.roi_tracker as rt

rt.ROITracker.load_prediction_history = lambda self, path=None: None

def test_prediction_summary():
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.7, workflow_id="wf1")
    tracker.record_prediction(1.0, 0.9, workflow_id="wf1")
    tracker.record_class_prediction("up", "up")
    tracker.record_class_prediction("down", "up")
    stats = tracker.prediction_summary()
    assert stats["mae"] == pytest.approx(0.15)
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["class_counts"]["predicted"]["up"] == 1
    assert stats["class_counts"]["actual"]["up"] == 2
    assert stats["confusion_matrix"]["up"]["up"] == 1
    assert stats["confusion_matrix"]["up"]["down"] == 1
    assert stats["mae_trend"][0] == pytest.approx(0.2)
    assert stats["mae_trend"][1] == pytest.approx(0.15)
    assert stats["accuracy_trend"][0] == pytest.approx(1.0)
    assert stats["accuracy_trend"][1] == pytest.approx(0.5)
    assert stats["scenario_roi_deltas"] == {}
    assert stats["worst_scenario"] is None
    wf_mae = stats["workflow_mae"]["wf1"]
    wf_var = stats["workflow_variance"]["wf1"]
    wf_conf = stats["workflow_confidence"]["wf1"]
    assert wf_mae[-1] == pytest.approx(tracker.workflow_mae("wf1"))
    assert wf_var[-1] == pytest.approx(tracker.workflow_variance("wf1"))
    assert wf_conf[-1] == pytest.approx(tracker.workflow_confidence("wf1"))

