import types
import sys
from pathlib import Path

class DummyBus:
    def __init__(self):
        self.subs = {}

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummyDataBot:
    def __init__(self, bus):
        self.event_bus = bus
        self.db = types.SimpleNamespace(fetch=lambda limit=50: [])

    def subscribe_degradation(self, cb):  # pragma: no cover - simple stub
        pass

    def reload_thresholds(self, _bot):
        return types.SimpleNamespace(error_threshold=0.1, roi_drop=-0.1)


class DummyCapital:
    trend_predictor = None


class DummyHistory:
    def add(self, *a, **k):
        pass


class DummyImprovement:
    pass


class DummyEvolutionManager:
    pass

def test_new_bot_event_triggers_subscription(tmp_path):
    bus = DummyBus()
    data_bot = DummyDataBot(bus)

    # lightweight stubs to avoid heavy imports
    data_bot_mod = types.ModuleType("menace_sandbox.data_bot")
    data_bot_mod.DataBot = object  # type: ignore
    sys.modules["menace_sandbox.data_bot"] = data_bot_mod

    capital_mod = types.ModuleType("menace_sandbox.capital_management_bot")
    capital_mod.CapitalManagementBot = DummyCapital
    sys.modules["menace_sandbox.capital_management_bot"] = capital_mod

    system_mod = types.ModuleType("menace_sandbox.system_evolution_manager")
    system_mod.SystemEvolutionManager = DummyEvolutionManager
    sys.modules["menace_sandbox.system_evolution_manager"] = system_mod

    history_mod = types.ModuleType("menace_sandbox.evolution_history_db")
    history_mod.EvolutionHistoryDB = DummyHistory
    history_mod.EvolutionEvent = object  # type: ignore
    sys.modules["menace_sandbox.evolution_history_db"] = history_mod

    eval_mod = types.ModuleType("menace_sandbox.evaluation_history_db")
    eval_mod.EvaluationHistoryDB = object  # type: ignore
    sys.modules["menace_sandbox.evaluation_history_db"] = eval_mod

    trend_mod = types.ModuleType("menace_sandbox.trend_predictor")
    trend_mod.TrendPredictor = object  # type: ignore
    sys.modules["menace_sandbox.trend_predictor"] = trend_mod

    threshold_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
    class _T:
        error_increase = 0.1
        roi_drop = -0.1
    threshold_mod.get_thresholds = lambda _bot=None: _T()
    sys.modules["menace_sandbox.self_coding_thresholds"] = threshold_mod

    scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")
    scm_mod.HelperGenerationError = Exception
    sys.modules["menace_sandbox.self_coding_manager"] = scm_mod

    ml_mod = types.ModuleType("menace_sandbox.mutation_logger")
    sys.modules["menace_sandbox.mutation_logger"] = ml_mod

    vec_mod = types.ModuleType("vector_service.context_builder")
    vec_mod.ContextBuilder = object  # type: ignore
    sys.modules["vector_service.context_builder"] = vec_mod

    from menace_sandbox.evolution_orchestrator import EvolutionOrchestrator

    orch = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolutionManager(),
        event_bus=bus,
        history_db=DummyHistory(),
        dataset_path=tmp_path / "ds.csv",
    )

    calls = []

    def fake_ensure(*_a, **_k):
        calls.append(1)

    orch._ensure_degradation_subscription = fake_ensure  # type: ignore

    orch.register_bot("bot1")
    assert len(calls) == 1

    bus.publish("bot:new", {"name": "bot2"})
    assert len(calls) == 2
