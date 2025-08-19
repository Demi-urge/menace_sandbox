import json
from pathlib import Path

from borderline_bucket import BorderlineBucket


def test_add_record_promote(tmp_path):
    path = tmp_path / "borderline_bucket.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("wf", 0.3, 0.8)
    bucket.record_result("wf", 0.5)
    bucket.promote("wf")

    assert bucket.candidates["wf"]["status"] == "promoted"
    assert bucket.candidates["wf"]["raroi"] == [0.3, 0.5]

    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert lines[-1]["action"] == "promote"
    assert len(lines) == 3


def test_terminate(tmp_path):
    path = tmp_path / "b.jsonl"
    bucket = BorderlineBucket(str(path))
    bucket.add_candidate("a", 0.1, 0.2)
    bucket.terminate("a")

    assert bucket.candidates["a"]["status"] == "terminated"

    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert lines[-1]["action"] == "terminate"
    assert len(lines) == 2
