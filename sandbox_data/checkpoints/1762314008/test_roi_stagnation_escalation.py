import ast
import importlib.util
import types
from pathlib import Path
from typing import Any, Dict

_BT_PATH = Path(__file__).resolve().parents[1] / "self_improvement" / "baseline_tracker.py"  # path-ignore
spec = importlib.util.spec_from_file_location("baseline_tracker", _BT_PATH)
baseline_tracker = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(baseline_tracker)  # type: ignore[attr-defined]
BaselineTracker = baseline_tracker.BaselineTracker

ENG_PATH = Path(__file__).resolve().parents[1] / "self_improvement" / "engine.py"  # path-ignore
src = ENG_PATH.read_text()
tree = ast.parse(src)
future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
sie_cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
)
method = next(
    n for n in sie_cls.body if isinstance(n, ast.FunctionDef)
    and n.name == "_check_roi_stagnation"
)
engine_module = ast.Module(
    [future, ast.ClassDef("SelfImprovementEngine", [], [], [method], [])],
    type_ignores=[],
)
engine_module = ast.fix_missing_locations(engine_module)
alerts: list[tuple] = []
ns: Dict[str, Any] = {
    "BaselineTracker": BaselineTracker,
    "dispatch_alert": lambda *a, **k: alerts.append((a, k)),
    "log_record": lambda **k: k,
}
exec(compile(engine_module, "<engine>", "exec"), ns)
SelfImprovementEngine = ns["SelfImprovementEngine"]


def _make_engine():
    class TrackerWrapper:
        def __init__(self) -> None:
            self.window = 5
            self._inner = BaselineTracker(window=self.window)

        def update(self, **metrics: float) -> None:
            self._inner.update(**metrics)

        def delta(self, metric: str) -> float:
            return self._inner.delta(metric)

        def std(self, metric: str) -> float:
            return self._inner.std(metric)

    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.baseline_tracker = TrackerWrapper()
    eng.roi_stagnation_dev_multiplier = 1.0
    eng.urgency_tier = 0
    eng.urgency_recovery_threshold = 0.05
    eng.logger = types.SimpleNamespace(warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    eng.stagnation_cycles = 3
    eng._roi_stagnation_count = 0
    return eng


def test_roi_stagnation_escalates():
    alerts.clear()
    eng = _make_engine()
    for roi in [1.0, 0.8, 0.6, 0.4]:
        eng.baseline_tracker.update(roi=roi)
        eng._check_roi_stagnation()
        assert eng.urgency_tier == 0
    eng.baseline_tracker.update(roi=0.2)
    eng._check_roi_stagnation()
    assert eng._roi_stagnation_count == 3
    assert eng.urgency_tier == 0
    eng.baseline_tracker.update(roi=0.0)
    eng._check_roi_stagnation()
    assert eng._roi_stagnation_count == 4
    assert eng.urgency_tier == 1
    assert alerts and alerts[0][0][0] == "roi_negative_trend"


def test_roi_recovery_resets():
    alerts.clear()
    eng = _make_engine()
    for roi in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        eng.baseline_tracker.update(roi=roi)
        eng._check_roi_stagnation()
    assert eng._roi_stagnation_count == 4
    assert eng.urgency_tier == 1
    eng.baseline_tracker.update(roi=1.0)
    eng._check_roi_stagnation()
    assert eng._roi_stagnation_count == 0
    assert eng.urgency_tier == 0
