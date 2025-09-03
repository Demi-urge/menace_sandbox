import asyncio
import threading
import ast
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence
import types


def _load_cycle_funcs() -> dict[str, Any]:
    src = Path("self_improvement/meta_planning.py").read_text()
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
        "BASELINE_TRACKER": types.SimpleNamespace(),
        "_get_entropy_threshold": lambda cfg, tracker: 1.0,
    }
    exec(compile(module, "<ast>", "exec"), ns)
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
    recent: tuple[Sequence[Any], float, int] | None = None,
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
                )
            ),
            "BASELINE_TRACKER": tracker,
        }
    )
    if recent is not None:
        meta["_recent_error_entropy"] = lambda *a, **k: recent

    async def _run():
        stop = threading.Event()
        task = asyncio.create_task(
            meta["self_improvement_cycle"]({"w": lambda: None}, interval=0, stop_event=stop)
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


def test_evaluate_cycle_skips_when_deltas_positive():
    meta = _load_cycle_funcs()
    tracker = DummyTracker({"roi": 1.0, "pass_rate": 0.5, "entropy": 0.1})
    record = {"timestamp": datetime.now().isoformat()}
    should_run, reason = meta["evaluate_cycle"](record, tracker, [])
    assert should_run is False
    assert reason == "all_deltas_positive"


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
        recent=(["trace"], 2.0, 1),
    )
    overfit = [rec for msg, rec in logs if rec.get("outcome") == "fallback"]
    assert any(r.get("reason") == "overfitting" for r in overfit)
