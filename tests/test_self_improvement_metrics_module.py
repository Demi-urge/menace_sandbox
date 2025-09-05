import pytest
from pathlib import Path

metrics = pytest.importorskip("menace.self_improvement.metrics")


def test_collect_metrics(tmp_path):
    sample = tmp_path / "sample.py"  # path-ignore
    sample.write_text("def f():\n    return 1\n")
    per_file, total, mi, test_count, ent, div = metrics._collect_metrics([sample], tmp_path)
    assert "sample.py" in per_file  # path-ignore
    assert total >= 0
    assert mi >= 0
    assert test_count == 0
    assert ent >= 0.0 and div >= 0.0
    assert "token_entropy" in per_file["sample.py"]  # path-ignore
    assert "token_diversity" in per_file["sample.py"]  # path-ignore
