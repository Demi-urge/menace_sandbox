import asyncio
import threading
import ast
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence
import types

import pytest
from dynamic_path_router import resolve_path


def _load_cycle_funcs() -> dict[str, Any]:
    src = resolve_path("self_improvement/meta_planning.py").read_text()
    tree = ast.parse(src)
    wanted = {
        "self_improvement_cycle",
        "evaluate_cycle",
        "_evaluate_cycle",
        "_recent_error_entropy",
    }
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
    module = ast.Module([future] + nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns: dict[str, Any] = {
        "asyncio": asyncio,
        "PLANNER_INTERVAL": 0.0,
        "SandboxSettings": object,
        "WorkflowStabilityDB": object,
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "TelemetryEvent": object,
        "datetime": datetime,
        "DEFAULT_SEVERITY_SCORE_MAP": {
            "critical": 100.0,
            "crit": 100.0,
            "fatal": 100.0,
            "high": 75.0,
            "error": 75.0,
            "warn": 50.0,
            "warning": 50.0,
            "medium": 50.0,
            "low": 25.0,
            "info": 0.0,
        },
        "BASELINE_TRACKER": types.SimpleNamespace(),
        "_get_entropy_threshold": lambda cfg, tracker: 1.0,
        "_init": types.SimpleNamespace(
            settings=types.SimpleNamespace(
                critical_severity_threshold=75.0,
                max_allowed_errors=0,
                entropy_overfit_threshold=1.0,
                error_window=5,
            )
        ),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    ns["REQUIRED_METRICS"] = ("roi", "pass_rate", "entropy")
    return ns


class DummyTracker:
    def __init__(self, deltas: Mapping[str, float]):
        self.deltas = dict(deltas)
        self._history = {k: [] for k in deltas}

    def update(self, **kw):
        pass

    def delta(self, metric: str) -> float:
        return float(self.deltas.get(metric, 0.0))

    def to_dict(self) -> Mapping[str, list[float]]:
        return {k: [v] for k, v in self.deltas.items()}

    def get(self, metric: str) -> float:
        return float(self.deltas.get(metric, 0.0))

    def std(self, metric: str) -> float:  # pragma: no cover - simple stub
        return 0.0

    @property
    def entropy_delta(self) -> float:
        return float(self.deltas.get("entropy", 0.0))


class DummyLogger:
    def __init__(self):
        self.records: list[tuple[str, Mapping[str, Any]]] = []
        self.errors: list[Exception] = []

    def debug(self, msg: str, *, extra: Mapping[str, Any] | None = None, **kw):
        self.records.append((msg, extra or {}))

    def exception(self, msg: str, *, exc_info=None, **kw):  # type: ignore[override]
        if exc_info:
            if isinstance(exc_info, tuple):
                self.errors.append(exc_info[1])
            else:
                self.errors.append(exc_info)

    info = warning = lambda *a, **k: None


def _run_cycle(
    deltas: Mapping[str, float],
    record: Mapping[str, Any],
    recent: tuple[Sequence[Any], float, int, float, float] | None = None,
    evaluator: Callable[[Any, Any | None], tuple[str, Mapping[str, Any]]] | None = None,
) -> list[tuple[str, Mapping[str, Any]]]:
    meta = _load_cycle_funcs()
    tracker = DummyTracker(deltas)
    logger = DummyLogger()

    class Planner:
        roi_db = None
        stability_db = None

        def __init__(self):
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            return [dict(record)]

    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "_FallbackPlanner": Planner,
            "get_logger": lambda name: logger,
            "log_record": lambda **kw: kw,
            "get_stable_workflows": lambda: types.SimpleNamespace(
                record_metrics=lambda *a, **k: None
            ),
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=0.0,
                    meta_roi_weight=0.0,
                    meta_domain_penalty=0.0,
                    overfitting_entropy_threshold=1.0,
                    entropy_overfit_threshold=1.0,
                    max_allowed_errors=0,
                    critical_severity_threshold=75.0,
                    error_window=5,
                )
            ),
            "BASELINE_TRACKER": tracker,
            "_get_overfit_thresholds": lambda cfg, _tracker: (
                getattr(cfg, "max_allowed_errors", 0),
                getattr(cfg, "entropy_overfit_threshold", 1.0),
            ),
        }
    )
    if recent is not None:
        meta["_recent_error_entropy"] = lambda *a, **k: recent

    async def _run():
        stop = threading.Event()
        kwargs = {"interval": 0, "stop_event": stop}
        if evaluator is not None:
            kwargs["evaluate_cycle"] = evaluator
        task = asyncio.create_task(
            meta["self_improvement_cycle"]({"w": lambda: None}, **kwargs)
        )
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 0.1)

    asyncio.run(_run())
    if logger.errors:
        raise logger.errors[0]
    return logger.records


def test_evaluate_cycle_runs_on_non_positive_delta():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": -0.1, "pass_rate": 1.0, "entropy": 0.0})
    record = {"timestamp": datetime.now().isoformat()}
    should_run, reason = meta["evaluate_cycle"](record, tracker, [])
    assert should_run is True
    assert reason == ""


def test_evaluate_cycle_runs_on_critical_error():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.0})
    now = datetime.now()
    record = {"timestamp": now.isoformat()}
    err = types.SimpleNamespace(
        error_type=types.SimpleNamespace(severity="critical"),
        timestamp=(now + timedelta(seconds=1)).isoformat(),
    )
    should_run, reason = meta["evaluate_cycle"](record, tracker, [err])
    assert should_run is True
    assert reason == ""


def test_evaluate_cycle_skips_with_noncritical_error():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1})
    now = datetime.now()
    record = {"timestamp": now.isoformat()}
    err = types.SimpleNamespace(
        error_type=types.SimpleNamespace(severity="warning"),
        timestamp=(now + timedelta(seconds=1)).isoformat(),
    )
    should_run, reason = meta["evaluate_cycle"](record, tracker, [err])
    assert should_run is False
    assert reason == "all_deltas_positive"


def test_evaluate_cycle_skips_when_deltas_positive():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 0.5, "entropy": 0.1})
    record = {"timestamp": datetime.now().isoformat()}
    should_run, reason = meta["evaluate_cycle"](record, tracker, [])
    assert should_run is False
    assert reason == "all_deltas_positive"


def test__evaluate_cycle_skips_when_required_deltas_positive():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1})
    decision, info = meta["_evaluate_cycle"](tracker, [])
    assert decision == "skip"
    assert info["reason"] == "all_deltas_positive"


@pytest.mark.parametrize("missing", ["roi", "pass_rate", "entropy"])
def test__evaluate_cycle_runs_when_metric_missing(missing):
    meta = _load_cycle_funcs()
    deltas = {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1}
    del deltas[missing]
    tracker = DummyTracker(deltas)
    decision, info = meta["_evaluate_cycle"](tracker, [])
    assert decision == "run"
    assert info["reason"] == "missing_metrics"
    assert missing in info["missing"]


@pytest.mark.parametrize("metric", ["roi", "pass_rate", "entropy"])
def test__evaluate_cycle_runs_on_non_positive_delta(metric):
    meta = _load_cycle_funcs()
    deltas = {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1}
    deltas[metric] = 0.0
    tracker = DummyTracker(deltas)
    decision, info = meta["_evaluate_cycle"](tracker, [])
    assert decision == "run"
    assert info["reason"] == "non_positive_delta"


@pytest.mark.parametrize(
    "severity",
    ["critical", "high", 0.9, "0.9", 9, "90"],
)
def test__evaluate_cycle_detects_critical_severity(severity):
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1})
    err = types.SimpleNamespace(error_type=types.SimpleNamespace(severity=severity))
    decision, info = meta["_evaluate_cycle"](tracker, [err])
    assert decision == "run"
    assert info["reason"] == "critical_error"


@pytest.mark.parametrize(
    "severity",
    ["medium", 0.2, "20"],
)
def test__evaluate_cycle_ignores_noncritical_severity(severity):
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1})
    err = types.SimpleNamespace(error_type=types.SimpleNamespace(severity=severity))
    decision, info = meta["_evaluate_cycle"](tracker, [err])
    assert decision == "skip"
    assert info["reason"] == "all_deltas_positive"


def test__evaluate_cycle_respects_severity_map_override():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1})
    meta["_init"].settings.severity_score_map = {"alert": 100.0}
    err = types.SimpleNamespace(error_type=types.SimpleNamespace(severity="alert"))
    decision, info = meta["_evaluate_cycle"](tracker, [err])
    assert decision == "run"
    assert info["reason"] == "critical_error"


def test_cycle_fallback_on_entropy_spike():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 5.0},
        {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 5.0},
    )
    assert logs[0][0] == "fallback"
    assert logs[0][1].get("reason") == "entropy_spike"


def test_cycle_fallback_on_error_traces():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.0},
        {"chain": ["w"], "roi_gain": 1.0, "failures": 1, "entropy": 0.0},
    )
    assert logs[0][0] == "fallback"
    assert logs[0][1].get("reason") == "errors_present"


def test_cycle_overfitting_fallback_logs_before_planner():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.0},
        {"chain": [], "roi_gain": 0.0, "failures": 0, "entropy": 0.0},
        recent=(["trace"], 0.2, 0, 0.05, 0.02),
    )
    overfit = [rec for msg, rec in logs if rec.get("outcome") == "fallback"]
    assert any(r.get("reason") == "overfitting" for r in overfit)


def test_cycle_reruns_on_recent_error_traces():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1},
        {"chain": [], "roi_gain": 0.0, "failures": 0, "entropy": 0.0},
        recent=(["trace"], 0.0, 2, 0.0, 1.0),
    )
    assert any(
        msg == "cycle" and rec.get("outcome") == "fallback" and rec.get("reason") == "overfitting"
        for msg, rec in logs
    )


def test_cycle_debug_logs_reason_for_positive_skip():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.1},
        {"chain": [], "roi_gain": 0.0, "failures": 0, "entropy": 0.0},
        recent=([], 0.0, 0, 0.0, 1.0),
    )
    assert any(
        msg == "cycle"
        and rec.get("outcome") == "skipped"
        and rec.get("reason") == "all_deltas_positive"
        for msg, rec in logs
    )


def test_cycle_logs_skip_on_stop_event():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({})
    logger = DummyLogger()

    class Planner:
        roi_db = None
        stability_db = None

        def __init__(self):
            self.cluster_map = {}

        def discover_and_persist(self, workflows):  # pragma: no cover - not used
            return []

    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "_FallbackPlanner": Planner,
            "get_logger": lambda name: logger,
            "log_record": lambda **kw: kw,
            "get_stable_workflows": lambda: types.SimpleNamespace(
                record_metrics=lambda *a, **k: None
            ),
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=0.0,
                    meta_roi_weight=0.0,
                    meta_domain_penalty=0.0,
                    overfitting_entropy_threshold=1.0,
                    entropy_overfit_threshold=1.0,
                    max_allowed_errors=0,
                    error_window=5,
                )
            ),
            "BASELINE_TRACKER": tracker,
        }
    )

    stop = threading.Event()
    stop.set()

    async def _run():
        await meta["self_improvement_cycle"](
            {"w": lambda: None}, interval=0, stop_event=stop
        )

    asyncio.run(_run())

    assert any(
        msg == "cycle"
        and rec.get("outcome") == "skipped"
        and rec.get("reason") == "stop_event"
        for msg, rec in logger.records
    )


def test_cycle_uses_provided_evaluator():
    calls: list[str] = []

    def custom_eval(tracker, errors):
        calls.append("called")
        return "skip", {"reason": "custom"}

    _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 0.0},
        {"chain": [], "roi_gain": 0.0, "failures": 0, "entropy": 0.0},
        evaluator=custom_eval,
    )
    assert calls


def test_cycle_requires_evaluator():
    meta = _load_cycle_funcs()

    async def _run():
        with pytest.raises(ValueError):
            await meta["self_improvement_cycle"](
                {"w": lambda: None}, interval=0, evaluate_cycle=None
            )

    asyncio.run(_run())
