import importlib.util
import roi_tracker as rt
from dynamic_path_router import resolve_path

spec = importlib.util.spec_from_file_location(
    "env", str(resolve_path("sandbox_runner/environment.py"))  # path-ignore
)
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)


def _write(path, roi=None, sec=None):
    t = rt.ROITracker()
    if roi is not None:
        t.metrics_history["synergy_roi"] = roi
    if sec is not None:
        t.metrics_history["synergy_security_score"] = sec
    t.save_history(str(path))


def test_synergy_ranking(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, roi=[0.2, 0.1], sec=[0.1])
    _write(b, roi=[-0.1], sec=[0.3])

    res = env.aggregate_synergy_metrics([str(a), str(b)])
    assert res[0][0] == "a"

    res_sec = env.aggregate_synergy_metrics([str(a), str(b)], metric="security_score")
    assert res_sec[0][0] == "b"
