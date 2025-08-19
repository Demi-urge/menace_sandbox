import json
from pathlib import Path

from borderline_bucket import BorderlineBucket


def test_add_record_promote(tmp_path):
    path = tmp_path / "borderline_bucket.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("wf", 0.3, 0.8)
    bucket.record_result("wf", 0.5)
    bucket.promote("wf")

    cand = bucket.get_candidate("wf")
    assert cand["status"] == "promoted"
    assert cand["raroi"] == [0.3, 0.5]

    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert lines[-1]["action"] == "promote"
    assert len(lines) == 3


def test_terminate(tmp_path):
    path = tmp_path / "b.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("a", 0.1, 0.2)
    bucket.terminate("a")

    cand = bucket.get_candidate("a")
    assert cand["status"] == "terminated"

    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert lines[-1]["action"] == "terminate"
    assert len(lines) == 2


def test_query_and_purge(tmp_path):
    path = tmp_path / "c.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("w1", 0.2, 0.6)
    assert bucket.get_candidate("w1") is not None
    bucket.promote("w1")
    bucket.purge("w1")
    assert bucket.get_candidate("w1") is None

    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert lines[-1]["action"] == "purge"
    assert len(lines) == 3
