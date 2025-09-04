from patch_safety import PatchSafety


def test_origin_specific_similarity_penalty():
    ps = PatchSafety(failure_db_path=None)
    err = {"category": "fail", "module": "m"}
    ps.record_failure(err, origin="error")
    ok_same, score_same, _ = ps.evaluate({}, err, origin="error")
    ok_other, score_other, _ = ps.evaluate({}, err, origin="other")
    assert not ok_same
    assert score_same > score_other
    assert score_same > 1.5
