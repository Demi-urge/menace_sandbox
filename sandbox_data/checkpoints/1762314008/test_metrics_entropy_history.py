import yaml
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

repo = Path(__file__).resolve().parents[1]
metrics_path = repo / "self_improvement" / ("metrics" + ".py")  # path-ignore
spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.metrics", metrics_path
)
metrics = importlib.util.module_from_spec(spec)
pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
pkg.__path__ = [str(repo)]  # type: ignore[attr-defined]
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)  # type: ignore[union-attr]

record_entropy = metrics.record_entropy
_update_alignment_baseline = metrics._update_alignment_baseline


def test_record_entropy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ("a" + ".py")).write_text("print('hi')\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    baseline = tmp_path / "baseline.yaml"
    settings = SimpleNamespace(alignment_baseline_metrics_path=baseline)
    record_entropy(0.2, 0.4, roi=1.0, settings=settings)
    record_entropy(0.4, 0.6, roi=0.5, settings=settings)
    data = yaml.safe_load(baseline.read_text())
    assert len(data["entropy_history"]) == 2
    assert data["entropy_history"][0]["roi"] == 1.0
    _update_alignment_baseline(settings)
    data2 = yaml.safe_load(baseline.read_text())
    assert "entropy_history" in data2 and len(data2["entropy_history"]) == 2
