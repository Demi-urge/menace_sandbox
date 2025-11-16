from patch_safety import PatchSafety


def test_metadata_rejection(tmp_path):
    ps = PatchSafety(max_alerts=1, max_alert_severity=0.5, storage_path=str(tmp_path / "f.jsonl"), failure_db_path=None)
    ok, score, _ = ps.evaluate({"license": "GPL-3.0"})
    assert not ok and score == 0.0
    ok, _, _ = ps.evaluate({"semantic_alerts": ["a", "b"]})
    assert not ok
    ok, _, _ = ps.evaluate({"alignment_severity": 0.75})
    assert not ok


def test_similarity_scoring(tmp_path):
    ps = PatchSafety(threshold=0.5, storage_path=str(tmp_path / "f.jsonl"), failure_db_path=None)
    failure = {"error": "boom"}
    ps.record_failure(failure)
    ok, score, _ = ps.evaluate({}, failure)
    assert not ok
    assert score >= ps.threshold
