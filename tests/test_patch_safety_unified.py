from patch_safety import PatchSafety


def test_metadata_rejection():
    ps = PatchSafety(max_alerts=1, max_alert_severity=0.5)
    ok, score = ps.evaluate({"license": "GPL-3.0"})
    assert not ok and score == 0.0
    ok, _ = ps.evaluate({"semantic_alerts": ["a", "b"]})
    assert not ok
    ok, _ = ps.evaluate({"alignment_severity": 0.75})
    assert not ok


def test_similarity_scoring():
    ps = PatchSafety(threshold=0.5)
    failure = {"error": "boom"}
    ps.record_failure(failure)
    ok, score = ps.evaluate({}, failure)
    assert not ok
    assert score >= ps.threshold
