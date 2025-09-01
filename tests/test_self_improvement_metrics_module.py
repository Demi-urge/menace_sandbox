import pytest
from pathlib import Path

metrics = pytest.importorskip("menace.self_improvement.metrics")


def test_collect_metrics(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("def f():\n    return 1\n")
    per_file, total, mi, test_count = metrics._collect_metrics([sample], tmp_path)
    assert "sample.py" in per_file
    assert total >= 0
    assert mi >= 0
    assert test_count == 0
