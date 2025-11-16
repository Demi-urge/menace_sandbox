from borderline_bucket import BorderlineBucket
from menace_sandbox.roi_tracker import ROITracker


def test_borderline_added_on_low_raroi(tmp_path):
    path = tmp_path / "b.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(raroi_borderline_threshold=0.1, borderline_bucket=bucket)
    tracker.workflow_confidence_scores["wf1"] = 0.9

    final, needs_review, conf = tracker.score_workflow("wf1", 0.05)

    assert not needs_review
    cand = bucket.get_candidate("wf1")
    assert cand is not None
    assert cand["raroi"] == [0.05]
    assert cand["confidence"] == 0.9


def test_borderline_added_on_low_confidence(tmp_path):
    path = tmp_path / "c.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(
        raroi_borderline_threshold=0.1,
        confidence_threshold=0.8,
        borderline_bucket=bucket,
    )
    tracker.workflow_confidence_scores["wf1"] = 0.5

    final, needs_review, conf = tracker.score_workflow("wf1", 0.2)

    assert needs_review
    cand = bucket.get_candidate("wf1")
    assert cand is not None
    assert cand["raroi"] == [0.2]
    assert cand["confidence"] == 0.5


def test_process_promotes_on_improved_raroi(tmp_path):
    path = tmp_path / "b.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(raroi_borderline_threshold=0.1, borderline_bucket=bucket)
    tracker.workflow_confidence_scores["wf1"] = 0.9
    tracker.score_workflow("wf1", 0.05)

    tracker.process_borderline_candidates(lambda wf, info: 0.2)

    cand = bucket.get_candidate("wf1")
    assert cand["status"] == "promoted"
    assert cand["raroi"][-1] == 0.2


def test_process_terminates_on_poor_raroi(tmp_path):
    path = tmp_path / "c.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(raroi_borderline_threshold=0.1, borderline_bucket=bucket)
    tracker.workflow_confidence_scores["wf1"] = 0.9
    tracker.score_workflow("wf1", 0.05)

    tracker.process_borderline_candidates(lambda wf, info: 0.01)

    cand = bucket.get_candidate("wf1")
    assert cand["status"] == "terminated"
    assert cand["raroi"][-1] == 0.01
