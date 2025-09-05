import asyncio
import sys
import types
import importlib.util
from pathlib import Path
import ast
import threading
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import pytest
from sandbox_settings import SandboxSettings
import self_improvement.baseline_tracker as baseline_tracker
from dynamic_path_router import resolve_path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _stub_deps():
    sys.modules.setdefault("quick_fix_engine", types.ModuleType("quick_fix_engine"))
    err_mod = types.ModuleType("error_logger")
    setattr(err_mod, "TelemetryEvent", type("TelemetryEvent", (), {}))
    sys.modules.setdefault("error_logger", err_mod)
    sys.modules.setdefault("menace.error_logger", err_mod)
    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    sys.modules.setdefault("sandbox_runner", sr_pkg)
    boot = types.ModuleType("sandbox_runner.bootstrap")
    boot.initialize_autonomous_sandbox = lambda *a, **k: None
    sys.modules.setdefault("sandbox_runner.bootstrap", boot)
    sys.modules.setdefault(
        "sandbox_runner.orphan_integration",
        types.ModuleType("sandbox_runner.orphan_integration"),
    )


def test_self_improvement_cycle_runs(tmp_path, monkeypatch, in_memory_dbs):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    InMemoryROI, InMemoryStability = in_memory_dbs
    roi_calls = {"count": 0}
    orig_log = InMemoryROI.log_result

    def spy(self, **kw):
        roi_calls["count"] += 1
        return orig_log(self, **kw)

    monkeypatch.setattr(InMemoryROI, "log_result", spy)
    baseline_calls: dict[str, int] = {}

    def track(metric: str) -> float:
        baseline_calls[metric] = baseline_calls.get(metric, 0) + 1
        return 0.0

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, *a, **k):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=DummyLock, Timeout=Exception, LOCK_TIMEOUT=1
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module

    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning",
        resolve_path("self_improvement/meta_planning.py"),  # path-ignore
    )

    monkeypatch.setattr(baseline_tracker.TRACKER, "get", track)
    monkeypatch.setattr(meta_planning.BASELINE_TRACKER, "get", track)

    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    calls = {"count": 0}

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROI()
            self.stability_db = InMemoryStability()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            calls["count"] += 1
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    monkeypatch.setattr(meta_planning, "_FallbackPlanner", DummyPlanner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    async def run_cycle():
        task = asyncio.create_task(
            meta_planning.self_improvement_cycle({"w": lambda: None}, interval=0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cycle())
    meta_planning.BASELINE_TRACKER.get("roi_delta")
    assert calls["count"] > 0
    assert roi_calls["count"] > 0
    assert InMemoryStability.instances[0].data
    assert baseline_calls.get("roi_delta", 0) > 0


def test_self_improvement_cycle_handles_db_errors(tmp_path, monkeypatch, in_memory_dbs):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    InMemoryROI, InMemoryStability = in_memory_dbs
    monkeypatch.setattr(
        InMemoryROI, "log_result", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, *a, **k):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=DummyLock, Timeout=Exception, LOCK_TIMEOUT=1
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module

    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning",
        resolve_path("self_improvement/meta_planning.py"),  # path-ignore
    )

    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROI()
            self.stability_db = InMemoryStability()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    monkeypatch.setattr(meta_planning, "_FallbackPlanner", DummyPlanner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    async def run_cycle():
        task = asyncio.create_task(
            meta_planning.self_improvement_cycle({"w": lambda: None}, interval=0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cycle())
    # error in ROI logging should not prevent stability metrics from recording
    assert InMemoryStability.instances[0].data


def test_start_self_improvement_cycle_dependency_failure(tmp_path, monkeypatch, in_memory_dbs):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    class BoomROI:
        def __init__(self, *a, **k):
            raise OSError("no db")

    # WorkflowStabilityDB may still be instantiated successfully
    InMemoryROI, InMemoryStability = in_memory_dbs

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False
        ),
        Timeout=Exception,
        LOCK_TIMEOUT=1,
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module
    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning",
        resolve_path("self_improvement/meta_planning.py"),  # path-ignore
    )
    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: SandboxSettings())
    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)
    init_module.init_self_improvement()

    monkeypatch.setattr(meta_planning, "ROIResultsDB", BoomROI)

    with pytest.raises(RuntimeError):
        meta_planning.start_self_improvement_cycle({"w": lambda: None}, interval=0)


# ---------------------------------------------------------------------------
# Additional behavioural tests for cycle evaluation logic


def _load_cycle_funcs():
    src = resolve_path("self_improvement/meta_planning.py").read_text()  # path-ignore
    tree = ast.parse(src)
    wanted = {
        "self_improvement_cycle",
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

    # unused log levels
    info = warning = lambda *a, **k: None


def _run_cycle(
    deltas: Mapping[str, float],
    record: Mapping[str, Any],
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


def test_cycle_runs_when_deltas_non_positive():
    logs = _run_cycle(
        {"roi": 0.0, "pass_rate": -0.1, "entropy": 0.0},
        {"chain": ["w"], "roi_gain": 0.0, "failures": 0, "entropy": 0.0},
    )
    rec = next(r for r in logs if r[1].get("outcome"))
    assert rec[1].get("outcome") == "skipped"
    assert rec[1].get("reason") == "no_delta"


def test_cycle_skips_when_deltas_positive():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 0.5, "entropy": 0.1},
        {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1},
    )
    rec = next(r for r in logs if r[1].get("outcome"))
    assert rec[1].get("outcome") == "skipped"
    assert rec[1].get("reason") == "all_deltas_positive"


def test_cycle_overfitting_fallback_on_entropy_spike():
    logs = _run_cycle(
        {"roi": 1.0, "pass_rate": 1.0, "entropy": 5.0},
        {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 5.0},
    )
    rec = next(r for r in logs if r[1].get("outcome"))
    assert rec[1].get("outcome") == "fallback"
    assert rec[1].get("reason") == "entropy_spike"
