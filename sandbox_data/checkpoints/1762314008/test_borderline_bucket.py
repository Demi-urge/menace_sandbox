import pytest
from unittest.mock import MagicMock

from borderline_bucket import BorderlineBucket
from menace_sandbox.roi_tracker import ROITracker


def test_enqueue_and_retrieve(tmp_path):
    path = tmp_path / "bucket.jsonl"
    bucket = BorderlineBucket(str(path))

    bucket.add_candidate("wf1", 0.1, 0.6, {"foo": "bar"})

    cand = bucket.get_candidate("wf1")
    assert cand is not None
    assert cand["raroi"] == [0.1]
    assert cand["confidence"] == 0.6
    assert cand["status"] == "pending"
    assert cand["context"] == {"foo": "bar"}


def test_record_result_appends_history(tmp_path):
    path = tmp_path / "bucket.jsonl"
    bucket = BorderlineBucket(str(path))

    bucket.add_candidate("wf1", 0.1, 0.8)
    bucket.record_result("wf1", 0.2, 0.9)
    bucket.record_result("wf1", 0.3)

    cand = bucket.get_candidate("wf1")
    assert cand["raroi"] == [0.1, 0.2, 0.3]
    assert cand["confidence"] == 0.9


def test_process_promotes_candidate(tmp_path):
    path = tmp_path / "bucket.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("good", 0.05, 0.7)

    def evaluator(wf, info):
        return 0.2, 0.8

    bucket.process(evaluator, raroi_threshold=0.1, confidence_threshold=0.6)

    cand = bucket.get_candidate("good")
    assert cand["status"] == "promoted"
    assert cand["raroi"][-1] == 0.2


def test_process_terminates_candidate(tmp_path):
    path = tmp_path / "bucket.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("bad", 0.05, 0.7)

    def evaluator(wf, info):
        return 0.01, 0.7

    bucket.process(evaluator, raroi_threshold=0.1, confidence_threshold=0.6)

    cand = bucket.get_candidate("bad")
    assert cand["status"] == "terminated"
    assert cand["raroi"][-1] == 0.01


def test_score_workflow_routes_borderline_cases(tmp_path):
    path = tmp_path / "bucket.jsonl"
    bucket = BorderlineBucket(str(path))
    tracker = ROITracker(
        raroi_borderline_threshold=0.1,
        confidence_threshold=0.6,
        borderline_bucket=bucket,
    )

    bucket.add_candidate = MagicMock()

    tracker.workflow_confidence_scores["low_raroi"] = 0.9
    tracker.workflow_confidence_scores["low_conf"] = 0.4

    tracker.score_workflow("low_raroi", 0.05)
    tracker.score_workflow("low_conf", 0.2)

    called_ids = {call.args[0] for call in bucket.add_candidate.call_args_list}
    assert called_ids == {"low_raroi", "low_conf"}
