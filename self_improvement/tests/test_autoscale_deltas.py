
import importlib.util
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path

MODULE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = MODULE_DIR.parent
sys.path.append(str(ROOT_DIR))

# Ensure package structure for relative imports
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(ROOT_DIR)]
sys.modules.setdefault("menace", menace_pkg)
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.__path__ = [str(MODULE_DIR)]
sys.modules.setdefault("menace.self_improvement", si_pkg)
sys.modules.setdefault("self_improvement", si_pkg)

# Stub minimal dependencies
log_records: list[tuple[str, str]] = []
logger = types.SimpleNamespace(
    warning=lambda msg, *a, **k: log_records.append(("warning", msg)),
    exception=lambda msg, *a, **k: log_records.append(("exception", msg)),
    info=lambda msg, *a, **k: log_records.append(("info", msg)),
    debug=lambda *a, **k: None,
)
logging_utils = types.SimpleNamespace(get_logger=lambda name: logger)
sys.modules["menace.logging_utils"] = logging_utils
sys.modules["logging_utils"] = logging_utils

model_module = types.ModuleType("menace.model_automation_pipeline")
model_module.AutomationResult = types.SimpleNamespace
sys.modules["menace.model_automation_pipeline"] = model_module
sys.modules["model_automation_pipeline"] = model_module

sandbox_runner = types.ModuleType("sandbox_runner")
sandbox_runner.bootstrap = types.SimpleNamespace(
    initialize_autonomous_sandbox=lambda *a, **k: None
)
sys.modules["sandbox_runner"] = sandbox_runner
sys.modules["sandbox_runner.bootstrap"] = sandbox_runner.bootstrap


class CapitalBot:
    def __init__(self) -> None:
        self.energy = 1.0

    def energy_score(self, **_: float) -> float:
        return self.energy

    def check_budget(self) -> bool:
        return True


capital_module = types.ModuleType("menace.capital_management_bot")
capital_module.CapitalManagementBot = CapitalBot
sys.modules["menace.capital_management_bot"] = capital_module
sys.modules["capital_management_bot"] = capital_module


class DataBot:
    def __init__(self) -> None:
        self.trend = 1.0

    def long_term_roi_trend(self, limit: int = 200) -> float:
        return self.trend


data_module = types.ModuleType("menace.data_bot")
data_module.DataBot = DataBot
sys.modules["menace.data_bot"] = data_module
sys.modules["data_bot"] = data_module


class DummyEngine:
    def __init__(self, name: str) -> None:
        self.name = name

    def _should_trigger(self) -> bool:  # pragma: no cover - not exercised
        return False

    def run_cycle(self, energy: int = 1):  # pragma: no cover - minimal stub
        return None

    def schedule(self, energy: int = 1, loop=None):  # pragma: no cover - minimal stub
        return types.SimpleNamespace()

    async def shutdown_schedule(self):  # pragma: no cover - minimal stub
        return None


engine_module = types.ModuleType("menace.self_improvement.engine")
engine_module.SelfImprovementEngine = DummyEngine
sys.modules["menace.self_improvement.engine"] = engine_module
sys.modules["self_improvement.engine"] = engine_module

settings = types.SimpleNamespace(
    roi=types.SimpleNamespace(deviation_tolerance=0.0),
    synergy=types.SimpleNamespace(deviation_tolerance=0.0),
)
init_module = types.ModuleType("menace.self_improvement.init")
init_module.settings = settings
sys.modules["menace.self_improvement.init"] = init_module
sys.modules["self_improvement.init"] = init_module

# Load baseline tracker and registry from source files
baseline_spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.baseline_tracker",
    resolve_path("self_improvement/baseline_tracker.py"),
)
baseline_module = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline_module)
sys.modules["menace.self_improvement.baseline_tracker"] = baseline_module
sys.modules["self_improvement.baseline_tracker"] = baseline_module
BaselineTracker = baseline_module.BaselineTracker

registry_spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.registry",
    resolve_path("self_improvement/registry.py"),
)
registry_module = importlib.util.module_from_spec(registry_spec)
registry_spec.loader.exec_module(registry_module)
ImprovementEngineRegistry = registry_module.ImprovementEngineRegistry


def _engine_factory(name: str) -> DummyEngine:
    return DummyEngine(name)


def _run_cycle(
    registry,
    tracker,
    cap,
    data,
    *,
    pass_rate,
    entropy,
    energy,
    roi,
    **kwargs,
):
    """Helper feeding metrics then invoking autoscale."""
    tracker.update(pass_rate=pass_rate, entropy=entropy)
    cap.energy = energy
    data.trend = roi
    registry.autoscale(
        capital_bot=cap,
        data_bot=data,
        factory=_engine_factory,
        **kwargs,
    )


def test_autoscale_responds_to_rolling_stats():
    tracker_low = BaselineTracker(window=5)
    baseline_module.TRACKER = tracker_low
    registry_module.BASELINE_TRACKER = tracker_low
    reg = ImprovementEngineRegistry()
    reg.register_engine("e0", DummyEngine("e0"))
    cap = CapitalBot()
    data = DataBot()

    for _ in range(5):
        _run_cycle(
            reg,
            tracker_low,
            cap,
            data,
            pass_rate=0.5,
            entropy=0.1,
            energy=1.0,
            roi=1.0,
            max_engines=1,
        )

    _run_cycle(
        reg,
        tracker_low,
        cap,
        data,
        pass_rate=0.55,
        entropy=0.08,
        energy=1.3,
        roi=1.3,
    )
    assert len(reg.engines) == 2

    tracker_high = BaselineTracker(window=5)
    baseline_module.TRACKER = tracker_high
    registry_module.BASELINE_TRACKER = tracker_high
    reg2 = ImprovementEngineRegistry()
    reg2.register_engine("e0", DummyEngine("e0"))
    cap2 = CapitalBot()
    data2 = DataBot()

    for energy, roi in [(0.5, 0.5), (1.5, 1.5), (0.5, 0.5), (1.5, 1.5), (1.0, 1.0)]:
        _run_cycle(
            reg2,
            tracker_high,
            cap2,
            data2,
            pass_rate=0.5,
            entropy=0.1,
            energy=energy,
            roi=roi,
            max_engines=1,
        )

    _run_cycle(
        reg2,
        tracker_high,
        cap2,
        data2,
        pass_rate=0.55,
        entropy=0.08,
        energy=1.3,
        roi=1.3,
    )
    assert len(reg2.engines) == 1


def test_urgency_escalation_after_negative_roi():
    tracker = BaselineTracker(window=5)
    baseline_module.TRACKER = tracker
    registry_module.BASELINE_TRACKER = tracker
    reg = ImprovementEngineRegistry()
    reg.register_engine("e0", DummyEngine("e0"))
    reg.register_engine("e1", DummyEngine("e1"))
    cap = CapitalBot()
    data = DataBot()

    for _ in range(3):
        _run_cycle(
            reg,
            tracker,
            cap,
            data,
            pass_rate=0.5,
            entropy=0.1,
            energy=1.0,
            roi=1.0,
            max_engines=2,
        )

    for i in range(4):
        _run_cycle(
            reg,
            tracker,
            cap,
            data,
            pass_rate=0.4,
            entropy=0.2,
            energy=1.0,
            roi=0.0,
            max_engines=2,
        )
        if i == 2:
            assert len(reg.engines) == 1
    assert len(reg.engines) == 1
