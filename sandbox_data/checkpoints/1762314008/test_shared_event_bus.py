import time
import sys
from pathlib import Path
import types
import contextvars

# Ensure repository root is on the Python path for direct module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Stub self_improvement package components
# ---------------------------------------------------------------------------
self_improv_pkg = types.ModuleType("menace_sandbox.self_improvement")
self_improv_pkg.__path__ = []  # type: ignore[attr-defined]

baseline_mod = types.ModuleType("menace_sandbox.self_improvement.baseline_tracker")


class BaselineTracker:  # pragma: no cover - simple stub
    def __init__(self, window: int = 5, metrics=None) -> None:
        self.window = window
        self._metrics = metrics or []

    def update(self, **kwargs) -> None:  # pragma: no cover - stub
        pass

    def std(self, metric: str) -> float:  # pragma: no cover - stub
        return 0.0


baseline_mod.BaselineTracker = BaselineTracker
sys.modules.setdefault("menace_sandbox.self_improvement", self_improv_pkg)
sys.modules.setdefault(
    "menace_sandbox.self_improvement.baseline_tracker", baseline_mod
)

target_mod = types.ModuleType("menace_sandbox.self_improvement.target_region")


class TargetRegion:  # pragma: no cover - simple stub
    pass


target_mod.TargetRegion = TargetRegion
sys.modules.setdefault("menace_sandbox.self_improvement.target_region", target_mod)

# ---------------------------------------------------------------------------
# Stub additional modules to avoid heavy dependencies
# ---------------------------------------------------------------------------
# sandbox_runner.test_harness
sandbox_pkg = types.ModuleType("menace_sandbox.sandbox_runner")
sandbox_pkg.__path__ = []  # type: ignore[attr-defined]
th_mod = types.ModuleType("menace_sandbox.sandbox_runner.test_harness")


class TestHarnessResult:  # pragma: no cover - simple stub
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok


def run_tests(*_a, **_k):  # pragma: no cover - stub
    return TestHarnessResult()


th_mod.run_tests = run_tests
th_mod.TestHarnessResult = TestHarnessResult
sys.modules.setdefault("menace_sandbox.sandbox_runner", sandbox_pkg)
sys.modules.setdefault("menace_sandbox.sandbox_runner.test_harness", th_mod)

# quick_fix_engine
qf_mod = types.ModuleType("menace_sandbox.quick_fix_engine")


class QuickFixEngine:  # pragma: no cover - simple stub
    def __init__(self, *a, **k):
        self.context_builder = k.get("context_builder")


def generate_patch(*a, **k):  # pragma: no cover - stub
    return ""


qf_mod.QuickFixEngine = QuickFixEngine
qf_mod.generate_patch = generate_patch
sys.modules.setdefault("menace_sandbox.quick_fix_engine", qf_mod)

# self_coding_engine
sce_mod = types.ModuleType("menace_sandbox.self_coding_engine")


class SelfCodingEngine:  # pragma: no cover - simple stub
    def __init__(self, *a, **k):
        builder = types.SimpleNamespace(
            refresh_db_weights=lambda: None, session_id=""
        )
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_suggestion_db = None
        self.patch_db = None


sce_mod.SelfCodingEngine = SelfCodingEngine
sce_mod.MANAGER_CONTEXT = contextvars.ContextVar("manager")
sys.modules.setdefault("menace_sandbox.self_coding_engine", sce_mod)

# model_automation_pipeline
map_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")


class ModelAutomationPipeline:  # pragma: no cover - simple stub
    pass


class AutomationResult:  # pragma: no cover - simple stub
    pass


map_mod.ModelAutomationPipeline = ModelAutomationPipeline
map_mod.AutomationResult = AutomationResult
sys.modules.setdefault("menace_sandbox.model_automation_pipeline", map_mod)

# error_bot
err_mod = types.ModuleType("menace_sandbox.error_bot")


class ErrorDB:  # pragma: no cover - simple stub
    pass


err_mod.ErrorDB = ErrorDB
sys.modules.setdefault("menace_sandbox.error_bot", err_mod)

# advanced_error_management and rollback_manager
aem_mod = types.ModuleType("menace_sandbox.advanced_error_management")


class FormalVerifier:  # pragma: no cover - simple stub
    def verify(self, _path=None):
        return True


class AutomatedRollbackManager:  # pragma: no cover - simple stub
    def log_healing_action(self, *a, **k):
        pass


aem_mod.FormalVerifier = FormalVerifier
#aaem_mod.AutomatedRollbackManager = AutomatedRollbackManager
aem_mod.AutomatedRollbackManager = AutomatedRollbackManager
sys.modules.setdefault("menace_sandbox.advanced_error_management", aem_mod)

rb_mod = types.ModuleType("menace_sandbox.rollback_manager")


class RollbackManager:  # pragma: no cover - simple stub
    pass


rb_mod.RollbackManager = RollbackManager
sys.modules.setdefault("menace_sandbox.rollback_manager", rb_mod)

# mutation_logger
mut_mod = types.ModuleType("menace_sandbox.mutation_logger")


def log_mutation(*a, **k):  # pragma: no cover - stub
    return 0


def set_event_bus(_b):  # pragma: no cover - stub
    pass


mut_mod.log_mutation = log_mutation
mut_mod.set_event_bus = set_event_bus
sys.modules.setdefault("menace_sandbox.mutation_logger", mut_mod)

cap_mod = types.ModuleType("menace_sandbox.capital_management_bot")


class CapitalManagementBot:  # pragma: no cover - simple stub
    def __init__(self, *a, **k):
        self.trend_predictor = None


cap_mod.CapitalManagementBot = CapitalManagementBot
sys.modules.setdefault("menace_sandbox.capital_management_bot", cap_mod)

sem_mod = types.ModuleType("menace_sandbox.system_evolution_manager")


class SystemEvolutionManager:  # pragma: no cover - simple stub
    def __init__(self, bots=None):
        self.bots = bots or []


sem_mod.SystemEvolutionManager = SystemEvolutionManager
sys.modules.setdefault("menace_sandbox.system_evolution_manager", sem_mod)

# ---------------------------------------------------------------------------
# Imports after stubbing
# ---------------------------------------------------------------------------
from menace_sandbox.shared_event_bus import event_bus
from menace_sandbox.data_bot import DataBot, MetricsDB
from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.self_coding_manager import SelfCodingManager
from menace_sandbox.evolution_orchestrator import EvolutionOrchestrator
from menace_sandbox.roi_thresholds import ROIThresholds


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


class DummyCognitionLayer:
    def __init__(self):
        self.context_builder = DummyContextBuilder()


class DummySelfCodingEngine:
    def __init__(self):
        self.cognition_layer = DummyCognitionLayer()
        self.patch_db = None
        self.patch_suggestion_db = None


class DummyPipeline:
    pass


class DummyQuickFixEngine:
    def __init__(self):
        self.context_builder = DummyContextBuilder()

    def validate_patch(self, *args, **kwargs):  # pragma: no cover - stub
        return True, []

    def apply_validated_patch(self, *args, **kwargs):  # pragma: no cover - stub
        return True, 0, []


class DummyCapitalBot:
    def __init__(self):
        self.trend_predictor = None


class DummyImprovementEngine:
    bot_name = "bot"


class DummyEvolutionManager:
    pass


def test_shared_event_bus_degradation_subscription():
    event_bus._subs.clear()
    event_bus._async_subs.clear()

    import tempfile
    tmp_db = MetricsDB(path=str(Path(tempfile.gettempdir()) / "metrics_test.db"))
    data_bot = DataBot(db=tmp_db, start_server=False, event_bus=event_bus)
    registry = BotRegistry(event_bus=event_bus)
    engine = DummySelfCodingEngine()
    pipeline = DummyPipeline()
    quick_fix = DummyQuickFixEngine()
    placeholder = object()
    manager = SelfCodingManager(
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        quick_fix=quick_fix,
        event_bus=event_bus,
        bot_name="bot",
        evolution_orchestrator=placeholder,
    )
    history_db = types.SimpleNamespace()
    orchestrator = EvolutionOrchestrator(
        data_bot=data_bot,
        capital_bot=DummyCapitalBot(),
        improvement_engine=DummyImprovementEngine(),
        evolution_manager=DummyEvolutionManager(),
        selfcoding_manager=manager,
        event_bus=event_bus,
        history_db=history_db,
    )
    manager.evolution_orchestrator = orchestrator

    assert data_bot.event_bus is event_bus
    assert registry.event_bus is event_bus
    assert manager.event_bus is event_bus
    assert orchestrator.event_bus is event_bus

    assert "degradation:detected" in event_bus._subs
    assert "bot:degraded" in event_bus._subs

    events = []
    event_bus.subscribe("degradation:detected", lambda _t, e: events.append(e))

    data_bot._thresholds["bot"] = ROIThresholds(-0.1, 0.1, 0.1)
    data_bot._last_threshold_refresh["bot"] = time.time()
    data_bot._baseline["bot"] = BaselineTracker(window=1)
    data_bot._ema_baseline["bot"] = {"roi": 0.0, "errors": 0.0, "tests_failed": 0.0}
    data_bot.baseline_window = 1
    data_bot.smoothing_factor = 1.0

    assert data_bot.check_degradation("bot", roi=-1.0, errors=2.0, test_failures=0.0)
    assert events and events[0]["bot"] == "bot"
