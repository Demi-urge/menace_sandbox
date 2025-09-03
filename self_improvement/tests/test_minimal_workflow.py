import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dynamic_path_router import resolve_path  # noqa: E402
from sandbox_settings import SandboxSettings  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


class InMemoryROIResultsDB:
    instances: list["InMemoryROIResultsDB"] = []

    def __init__(self, *a, **k):
        self.records: list[types.SimpleNamespace] = []
        self.__class__.instances.append(self)

    def log_result(self, **kwargs):
        self.records.append(types.SimpleNamespace(**kwargs))


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


def test_minimal_workflow(tmp_path, monkeypatch):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    sys.modules["sandbox_runner"] = sr_pkg
    # stub external dependencies
    sys.modules.setdefault("quick_fix_engine", types.ModuleType("quick_fix_engine"))
    sys.modules.setdefault("sandbox_runner.orphan_integration", types.ModuleType("sandbox_runner.orphan_integration"))
    sys.modules.setdefault("relevancy_radar", types.ModuleType("relevancy_radar"))
    sys.modules.setdefault("error_logger", types.ModuleType("error_logger"))
    sys.modules.setdefault("telemetry_feedback", types.ModuleType("telemetry_feedback"))
    sys.modules.setdefault("telemetry_backend", types.ModuleType("telemetry_backend"))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault("sandbox_runner.environment", types.ModuleType("sandbox_runner.environment"))

    log_messages: list[tuple[str, tuple, dict]] = []
    logger = types.SimpleNamespace(
        info=lambda *a, **k: log_messages.append(("info", a, k)),
        warning=lambda *a, **k: log_messages.append(("warning", a, k)),
        exception=lambda *a, **k: log_messages.append(("exception", a, k)),
        debug=lambda *a, **k: log_messages.append(("debug", a, k)),
        error=lambda *a, **k: log_messages.append(("error", a, k)),
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    init_calls = {"count": 0}

    def fake_init(settings):
        init_calls["count"] += 1

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = fake_init
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

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
    monkeypatch.setattr(
        patch_generation, "_call_with_retries", lambda f, *a, **k: f(*a, **k)
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_repo_path = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False
    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(init_module, "verify_dependencies", lambda: None)
    init_module.init_self_improvement()
    synergy_path = Path(settings.synergy_weight_file)
    before = json.loads(synergy_path.read_text())

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROIResultsDB()
            self.stability_db = InMemoryWorkflowStabilityDB()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            data = json.loads(synergy_path.read_text())
            data["roi"] += 1
            synergy_path.write_text(json.dumps(data))
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    monkeypatch.setattr(meta_planning, "_FallbackPlanner", DummyPlanner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    bus = types.SimpleNamespace(publish=lambda *a, **k: None)
    thread = meta_planning.start_self_improvement_cycle(
        {"w": lambda: None}, event_bus=bus, interval=0
    )
    thread.start()
    time.sleep(0.05)
    thread.stop()

    after = json.loads(synergy_path.read_text())
    assert init_calls["count"] == 1
    assert InMemoryROIResultsDB.instances[0].records
    assert InMemoryWorkflowStabilityDB.instances[0].data
    assert log_messages
    assert after["roi"] == before["roi"] + 1
