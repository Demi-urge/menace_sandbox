import ast
import importlib.util
import types
import time
from typing import Any, Dict

from dynamic_path_router import resolve_path

# Load BaselineTracker
BT_SPEC = importlib.util.spec_from_file_location(
    "baseline_tracker", resolve_path("self_improvement/baseline_tracker.py")
)
baseline_mod = importlib.util.module_from_spec(BT_SPEC)
assert BT_SPEC and BT_SPEC.loader
BT_SPEC.loader.exec_module(baseline_mod)  # type: ignore[attr-defined]
BaselineTracker = baseline_mod.BaselineTracker

# Extract selected methods from engine.py
ENG_PATH = resolve_path("self_improvement/engine.py")
engine_src = ENG_PATH.read_text()
engine_tree = ast.parse(engine_src)
future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
class_def = next(
    n for n in engine_tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
)
should_trigger = next(
    n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "_should_trigger"
)
check_stagnation = next(
    n
    for n in class_def.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "_check_roi_stagnation"
)
engine_module = ast.Module(
    [future, ast.ClassDef("SelfImprovementEngine", [], [], [should_trigger, check_stagnation], [])],
    type_ignores=[],
)
engine_module = ast.fix_missing_locations(engine_module)
alerts: list[tuple] = []
ns: Dict[str, Any] = {
    "time": time,
    "log_record": lambda **k: k,
    "dispatch_alert": lambda *a, **k: alerts.append((a, k)),
    "settings": types.SimpleNamespace(critical_severity_threshold=75.0),
}
exec(compile(engine_module, "<engine>", "exec"), ns)
SelfImprovementEngine = ns["SelfImprovementEngine"]

# Extract _should_encode from meta_planning.py
MP_PATH = resolve_path("self_improvement/meta_planning.py")
mp_src = MP_PATH.read_text()
mp_tree = ast.parse(mp_src)
should_encode_func = next(
    n for n in mp_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_should_encode"
)
mp_module = ast.Module([future, should_encode_func], type_ignores=[])
mp_module = ast.fix_missing_locations(mp_module)
mp_ns: Dict[str, Any] = {}
exec(compile(mp_module, "<meta>", "exec"), mp_ns)
_should_encode = mp_ns["_should_encode"]


def test_should_trigger_skips_only_on_positive_deltas_and_no_critical_errors():
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0, pass_rate=1.0, record_momentum=False)
    tracker.update(roi=1.2, pass_rate=1.1, record_momentum=False)
    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.baseline_tracker = tracker
    eng.error_bot = types.SimpleNamespace(recent_errors=lambda limit=5: [])
    eng.last_run = 0.0
    eng.interval = 0.0
    eng.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    assert not eng._should_trigger()

    tracker.update(roi=0.5, pass_rate=1.0, record_momentum=False)
    assert eng._should_trigger()

    tracker.update(roi=1.5, pass_rate=1.2, record_momentum=False)
    err = types.SimpleNamespace(error_type=types.SimpleNamespace(severity="critical"))
    eng.error_bot = types.SimpleNamespace(recent_errors=lambda limit=5: [err])
    assert eng._should_trigger()


def test_should_encode_uses_roi_delta_and_momentum():
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0, pass_rate=1.0, entropy=0.4)
    record = {"roi_gain": 1.2, "entropy": 0.4, "failures": 0}
    tracker.update(roi=record["roi_gain"], pass_rate=1.0, entropy=record["entropy"])
    ok, reason = _should_encode(record, tracker, entropy_threshold=0.5)
    assert ok and reason == "improved"

    record_neg = {"roi_gain": 0.8, "entropy": 0.4, "failures": 0}
    tracker_neg = BaselineTracker(window=3)
    tracker_neg.update(roi=1.0, pass_rate=1.0, entropy=0.4)
    tracker_neg.update(
        roi=record_neg["roi_gain"], pass_rate=1.0, entropy=record_neg["entropy"]
    )
    ok_neg, reason_neg = _should_encode(record_neg, tracker_neg, entropy_threshold=0.5)
    assert not ok_neg and reason_neg == "no_delta"

    tracker_zero = BaselineTracker(window=3)
    tracker_zero.update(roi=1.0, pass_rate=1.0, entropy=0.4)
    tracker_zero._success_history.clear()
    tracker_zero._success_history.extend([False] * tracker_zero.window)
    tracker_zero.update(roi=record["roi_gain"], pass_rate=1.0, entropy=record["entropy"])
    assert tracker_zero.momentum == 0.0
    ok_zero, _ = _should_encode(record, tracker_zero, entropy_threshold=0.5)
    assert not ok_zero


def test_urgency_escalates_after_persistent_roi_decline():
    alerts.clear()
    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.baseline_tracker = BaselineTracker(window=5)
    eng.roi_stagnation_dev_multiplier = 1.0
    eng.urgency_tier = 0
    eng.urgency_recovery_threshold = 0.05
    eng.logger = types.SimpleNamespace(warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    eng.stagnation_cycles = 3
    eng._roi_stagnation_count = 0
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
