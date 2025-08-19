from borderline_bucket import BorderlineBucket
from menace_sandbox.roi_tracker import ROITracker


def test_borderline_added_on_low_raroi(tmp_path):
    path = tmp_path / "b.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(borderline_threshold=0.1, borderline_bucket=bucket)
    tracker.workflow_confidence_scores["wf1"] = 0.9

    final, needs_review, conf = tracker.score_workflow("wf1", 0.05)

    assert not needs_review
    cand = bucket.get_candidate("wf1")
    assert cand is not None
    assert cand["raroi"] == [0.05]
    assert cand["confidence"] == 0.9
