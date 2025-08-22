import json
import time

from patch_safety import PatchSafety


def test_record_failure_refresh(tmp_path):
    store = tmp_path / "failures.jsonl"
    ps = PatchSafety(storage_path=str(store), failure_db_path=None, refresh_interval=0.1)
    err_existing = {"category": "fail", "module": "m"}
    with store.open("w", encoding="utf-8") as fh:
        json.dump({"err": err_existing}, fh)
        fh.write("\n")
    ok, _ = ps.evaluate({}, err_existing)
    assert ok
    time.sleep(0.15)
    ps.record_failure({"category": "other", "module": "m"})
    assert len(ps._records) == 2
    ok, score = ps.evaluate({}, err_existing)
    assert not ok
    assert score >= ps.threshold
