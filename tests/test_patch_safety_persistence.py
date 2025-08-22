from patch_safety import PatchSafety


def test_persistence_roundtrip(tmp_path):
    store = tmp_path / "failures.jsonl"
    ps = PatchSafety(storage_path=str(store))
    failure = {"category": "fail", "module": "m.py"}
    ps.record_failure(failure)
    assert store.exists()
    ps2 = PatchSafety(storage_path=str(store))
    ok, score, _ = ps2.evaluate({}, failure)
    assert not ok
    assert score >= ps.threshold
