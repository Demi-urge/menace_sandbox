import pytest

from menace_sandbox.borderline_bucket import BorderlineBucket


@pytest.fixture()
def bucket(tmp_path):
    path = tmp_path / "bucket.jsonl"
    return BorderlineBucket(str(path))


def test_enqueue_and_pending(bucket):
    bucket.enqueue("wf1", 0.1, 0.9, {"foo": "bar"})
    bucket.enqueue("wf2", 0.2, 0.8, None)

    pending = bucket.pending()

    assert set(pending.keys()) == {"wf1", "wf2"}
    assert pending["wf1"]["context"] == {"foo": "bar"}
    assert bucket.status("wf1") == "candidate"


def test_record_outcome_and_promote(bucket):
    bucket.enqueue("wf1", 0.1, 0.9, {})
    bucket.record_result("wf1", 0.15, 0.95)

    cand = bucket.get_candidate("wf1")
    assert cand["raroi"] == [0.1, 0.15]
    assert cand["confidence"] == 0.95

    bucket.promote("wf1")
    assert bucket.status("wf1") == "promoted"


def test_terminate(bucket):
    bucket.enqueue("wf1", 0.1, 0.9, None)
    bucket.terminate("wf1")

    assert bucket.status("wf1") == "terminated"


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "bucket.jsonl"
    b1 = BorderlineBucket(str(path))
    b1.enqueue("wf1", 0.1, 0.9, {"a": 1})
    b1.promote("wf1")

    b2 = BorderlineBucket(str(path))
    cand = b2.get_candidate("wf1")
    assert cand == {
        "raroi": [0.1],
        "confidence": 0.9,
        "status": "promoted",
        "context": {"a": 1},
    }


def test_process_promotes_and_terminates(bucket):
    bucket.enqueue("good", 0.05, 0.8)
    bucket.enqueue("bad", 0.05, 0.8)

    def evaluator(wf, info):
        if wf == "good":
            return 0.2, 0.85
        return 0.01, 0.4

    bucket.process(
        evaluator, raroi_threshold=0.1, confidence_threshold=0.5
    )

    assert bucket.status("good") == "promoted"
    assert bucket.status("bad") == "terminated"
    assert bucket.get_candidate("good")["confidence"] == 0.85

