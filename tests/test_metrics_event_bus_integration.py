import sys
import types

from menace.unified_event_bus import UnifiedEventBus
from menace.data_bot import DataBot, MetricsDB, ROIThresholds


def test_event_bus_propagates_degradation(tmp_path, monkeypatch):
    stub_cbi = types.ModuleType("menace.coding_bot_interface")
    stub_cbi.self_coding_managed = lambda cls: cls
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", stub_cbi)

    stub_cap = types.ModuleType("menace.capital_management_bot")
    class CapitalManagementBot:  # minimal stub
        trend_predictor = None
    stub_cap.CapitalManagementBot = CapitalManagementBot
    monkeypatch.setitem(sys.modules, "menace.capital_management_bot", stub_cap)

    stub_sem = types.ModuleType("menace.system_evolution_manager")
    class SystemEvolutionManager:  # minimal stub
        pass
    stub_sem.SystemEvolutionManager = SystemEvolutionManager
    monkeypatch.setitem(sys.modules, "menace.system_evolution_manager", stub_sem)

    stub_scm = types.ModuleType("menace.self_coding_manager")
    stub_scm.HelperGenerationError = RuntimeError
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", stub_scm)

    stub_history = types.ModuleType("menace.evolution_history_db")
    stub_history.EvolutionHistoryDB = type("EvolutionHistoryDB", (), {})
    stub_history.EvolutionEvent = type("EvolutionEvent", (), {})
    monkeypatch.setitem(sys.modules, "menace.evolution_history_db", stub_history)

    stub_eval = types.ModuleType("menace.evaluation_history_db")
    stub_eval.EvaluationHistoryDB = type("EvaluationHistoryDB", (), {})
    monkeypatch.setitem(sys.modules, "menace.evaluation_history_db", stub_eval)

    stub_trend = types.ModuleType("menace.trend_predictor")
    stub_trend.TrendPredictor = type("TrendPredictor", (), {})
    monkeypatch.setitem(sys.modules, "menace.trend_predictor", stub_trend)

    stub_thresh = types.ModuleType("menace.self_coding_thresholds")
    def get_thresholds(_bot=None):
        return types.SimpleNamespace(error_increase=0.1, roi_drop=-0.1)
    stub_thresh.get_thresholds = get_thresholds
    monkeypatch.setitem(sys.modules, "menace.self_coding_thresholds", stub_thresh)

    stub_mut = types.ModuleType("menace.mutation_logger")
    monkeypatch.setitem(sys.modules, "menace.mutation_logger", stub_mut)

    stub_ctx = types.ModuleType("vector_service.context_builder")
    stub_ctx.ContextBuilder = type("ContextBuilder", (), {})
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", stub_ctx)

    from menace.evolution_orchestrator import EvolutionOrchestrator, EvolutionTrigger

    bus = UnifiedEventBus()
    db = MetricsDB(tmp_path / "m.db")
    data_bot = DataBot(db=db, start_server=False, event_bus=bus)
    data_bot._thresholds["b1"] = ROIThresholds(
        roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0
    )

    metrics_events: list[dict] = []
    degradation_events: list[dict] = []
    bus.subscribe("metrics:updated", lambda t, e: metrics_events.append(e))
    bus.subscribe("degradation:detected", lambda t, e: degradation_events.append(e))

    class DummyImprovement:
        pass

    class DummySelfCodingManager:
        def __init__(self) -> None:
            self.bot_name = "b1"
            self.events: list[dict] = []

        def register_patch_cycle(self, event: dict) -> None:
            self.events.append(event)

    scm = DummySelfCodingManager()

    EvolutionOrchestrator(
        data_bot=data_bot,
        capital_bot=CapitalManagementBot(),
        improvement_engine=DummyImprovement(),
        evolution_manager=SystemEvolutionManager(),
        selfcoding_manager=scm,
        triggers=EvolutionTrigger(error_rate=1.0, roi_drop=-0.1),
        event_bus=bus,
    )

    data_bot.check_degradation("b1", roi=1.0, errors=0.0)
    data_bot.check_degradation("b1", roi=0.0, errors=2.0)

    assert metrics_events and metrics_events[-1]["bot"] == "b1"
    assert degradation_events and degradation_events[-1]["bot"] == "b1"
    assert scm.events and scm.events[-1]["bot"] == "b1"

