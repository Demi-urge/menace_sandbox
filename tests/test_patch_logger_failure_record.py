from patch_safety import PatchSafety
from vector_service.patch_logger import PatchLogger


def test_track_contributors_records_failure(tmp_path):
    store = tmp_path / "failures.jsonl"
    ps = PatchSafety(storage_path=str(store), failure_db_path=None)
    pl = PatchLogger(patch_safety=ps)
    meta = {"error:1": {"category": "fail", "module": "m"}}
    pl.track_contributors(["error:1"], False, retrieval_metadata=meta)
    ps2 = PatchSafety(storage_path=str(store), failure_db_path=None)
    ok, score, _ = ps2.evaluate({}, {"category": "fail", "module": "m"})
    assert not ok
    assert score >= ps.threshold
