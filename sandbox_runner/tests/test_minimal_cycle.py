import importlib.util
import sys
import time
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.path.append(str(resolve_path("")))

from sandbox_settings import SandboxSettings  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _setup_base_packages():
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
    sys.modules["menace.error_logger"] = types.SimpleNamespace(
        TelemetryEvent=object
    )


class InMemoryROIResultsDB:
    instances: list["InMemoryROIResultsDB"] = []

    def __init__(self, *a, **k):
        self.records: list[types.SimpleNamespace] = []
        self.__class__.instances.append(self)

    def log_result(self, **kwargs):
        self.records.append(types.SimpleNamespace(**kwargs))

    def fetch_results(self, workflow_id: str, run_id: str | None = None):
        return [
            r
            for r in self.records
            if r.workflow_id == workflow_id and (run_id is None or r.run_id == run_id)
        ]


class InMemoryWorkflowStabilityDB:
    instances: list["InMemoryWorkflowStabilityDB"] = []

    def __init__(self, *a, **k):
        self.data: dict[str, dict[str, float]] = {}
        self.__class__.instances.append(self)

    def record_metrics(
        self,
        workflow_id: str,
        roi: float,
        failures: float,
        entropy: float,
        *,
        roi_delta: float | None = None,
        roi_var: float = 0.0,
        failures_var: float = 0.0,
        entropy_var: float = 0.0,
    ) -> None:
        prev = self.data.get(workflow_id, {}).get("roi", 0.0)
        delta = roi - prev if roi_delta is None else roi_delta
        self.data[workflow_id] = {
            "roi": roi,
            "roi_delta": delta,
            "roi_var": roi_var,
            "failures": failures,
            "failures_var": failures_var,
            "entropy": entropy,
            "entropy_var": entropy_var,
        }

    def is_stable(
        self, workflow_id: str, current_roi: float | None = None, threshold: float | None = None
    ) -> bool:
        return True


def test_cycle_generates_patch_and_metrics(tmp_path, monkeypatch):
    _setup_base_packages()
    import sandbox_settings as sandbox_settings_module
    sys.modules["menace.sandbox_settings"] = sandbox_settings_module
    sys.modules["menace.roi_results_db"] = types.SimpleNamespace(
        ROIResultsDB=InMemoryROIResultsDB
    )
    sys.modules["menace.workflow_stability_db"] = types.SimpleNamespace(
        WorkflowStabilityDB=InMemoryWorkflowStabilityDB
    )
    sys.modules.setdefault("quick_fix_engine", types.ModuleType("quick_fix_engine"))
    sys.modules.setdefault(
        "menace.quick_fix_engine", sys.modules["quick_fix_engine"]
    )
    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False
        ),
        Timeout=Exception,
        LOCK_TIMEOUT=1,
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(MetaWorkflowPlanner=None)

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")
    )
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning",
        resolve_path("self_improvement/meta_planning.py"),
    )
    patch_generation = _load_module(
        "menace.self_improvement.patch_generation",
        resolve_path("self_improvement/patch_generation.py"),
    )

    monkeypatch.setattr(meta_planning, "ROIResultsDB", InMemoryROIResultsDB)
    monkeypatch.setattr(meta_planning, "WorkflowStabilityDB", InMemoryWorkflowStabilityDB)
    monkeypatch.setattr(patch_generation, "_load_callable", lambda m, n: lambda *a, **k: 1)
    monkeypatch.setattr(patch_generation, "_call_with_retries", lambda f, *a, **k: f(*a, **k))

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_repo_path = str(tmp_path)
    settings.sandbox_central_logging = False
    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROIResultsDB()
            self.stability_db = InMemoryWorkflowStabilityDB()
            self.cluster_map = {}
            self.patches: list[int] = []
            self._called = False

        def discover_and_persist(self, workflows):
            self.patches.append(
                patch_generation.generate_patch(context_builder=object())
            )
            if self._called:
                return []
            self._called = True
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    planner = DummyPlanner()
    monkeypatch.setattr(meta_planning, "_FallbackPlanner", lambda: planner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    class Bus:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []

        def publish(self, event, payload):
            self.events.append((event, payload))

    bus = Bus()
    thread = meta_planning.start_self_improvement_cycle(
        {"w": lambda: None}, event_bus=bus, interval=0
    )
    thread.start()
    time.sleep(0.05)
    thread.stop()

    roi_db = planner.roi_db
    stability_db = planner.stability_db
    assert roi_db.records
    assert stability_db.data
    assert planner.patches
    assert bus.events and bus.events[0][0] == "metrics:new"
