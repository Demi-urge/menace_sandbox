import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

md_mod = _load("metrics_dashboard")
ROITracker = _load("roi_tracker").ROITracker
MetricsDashboard = md_mod.MetricsDashboard


def _make_history(path: Path) -> None:
    tracker = ROITracker()
    tracker.record_prediction(1.0, 1.5)
    tracker.record_prediction(2.0, 1.0)
    tracker.category_history.extend(["good", "bad", "good"])
    tracker.save_history(str(path))


def test_new_plots(tmp_path):
    hist = tmp_path / "hist.json"
    _make_history(hist)
    dash = MetricsDashboard(hist)
    client = dash.app.test_client()
    for ep in [
        "/plots/prediction_error.png",
        "/plots/roi_category_distribution.png",
        "/plots/roi_improvement.png",
    ]:
        resp = client.get(ep)
        assert resp.status_code == 200
        assert resp.data.startswith(b"\x89PNG") or resp.data == b""
