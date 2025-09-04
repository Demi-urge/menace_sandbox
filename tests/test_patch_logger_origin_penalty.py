from patch_safety import PatchSafety
from vector_service.patch_logger import PatchLogger


def test_patch_logger_origin_penalty():
    ps = PatchSafety(threshold=10.0, failure_db_path=None)
    pl = PatchLogger(patch_safety=ps)
    failure_meta = {"error:1": {"category": "fail", "module": "m"}}
    pl.track_contributors(["error:1"], False, retrieval_metadata=failure_meta)

    meta_error = {"error:2": {"category": "fail", "module": "m"}}
    scores_error = pl.track_contributors(["error:2"], True, retrieval_metadata=meta_error)
    assert scores_error["error"] > 1.5

    meta_info = {"information:1": {"category": "fail", "module": "m"}}
    scores_info = pl.track_contributors(["information:1"], True, retrieval_metadata=meta_info)
    assert scores_error["error"] > scores_info["information"]
