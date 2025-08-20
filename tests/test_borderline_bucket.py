import pytest

from menace_sandbox.borderline_bucket import BorderlineBucket


@pytest.fixture()
def bucket(tmp_path):
    path = tmp_path / "bucket.jsonl"
    return BorderlineBucket(str(path))


def test_add_and_list_candidates(bucket):
    bucket.add_candidate("wf1", 0.1, 0.9)
    bucket.add_candidate("wf2", 0.2, 0.8)

    all_cands = bucket.all_candidates()

    assert set(all_cands.keys()) == {"wf1", "wf2"}
    assert all_cands["wf1"]["raroi"] == [0.1]
    assert all_cands["wf1"]["status"] == "candidate"


def test_record_outcome_and_promote(bucket):
    bucket.add_candidate("wf1", 0.1, 0.9)
    bucket.record_result("wf1", 0.15)

    cand = bucket.get_candidate("wf1")
    assert cand["raroi"] == [0.1, 0.15]

    bucket.promote("wf1")
    assert bucket.get_candidate("wf1")["status"] == "promoted"


def test_terminate(bucket):
    bucket.add_candidate("wf1", 0.1, 0.9)
    bucket.terminate("wf1")

    cand = bucket.get_candidate("wf1")
    assert cand["status"] == "terminated"


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "bucket.jsonl"
    b1 = BorderlineBucket(str(path))
    b1.add_candidate("wf1", 0.1, 0.9)
    b1.promote("wf1")

    b2 = BorderlineBucket(str(path))
    cand = b2.get_candidate("wf1")
    assert cand == {
        "raroi": [0.1],
        "confidence": 0.9,
        "status": "promoted",
    }

