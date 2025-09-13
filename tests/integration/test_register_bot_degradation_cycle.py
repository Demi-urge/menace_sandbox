import sys
import types
from pathlib import Path


def test_degradation_event_registers_patch_cycle(tmp_path, monkeypatch):
    ROOT = Path(__file__).resolve().parents[2]
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    sys.modules.setdefault("menace", pkg)

    vec_cb = types.ModuleType("vector_service.context_builder")

    class ContextBuilder:
        def refresh_db_weights(self):
            pass

    vec_cb.ContextBuilder = ContextBuilder
    vec_cb.record_failed_tags = lambda *a, **k: None
    vec_cb.load_failed_tags = lambda *a, **k: set()
    sys.modules.setdefault("vector_service.context_builder", vec_cb)

    cap_mod = types.ModuleType("menace.capital_management_bot")
    cap_mod.CapitalManagementBot = types.SimpleNamespace
    sys.modules.setdefault("menace.capital_management_bot", cap_mod)
    sem_mod = types.ModuleType("menace.system_evolution_manager")
    sem_mod.SystemEvolutionManager = types.SimpleNamespace
    sys.modules.setdefault("menace.system_evolution_manager", sem_mod)
    hist_mod = types.ModuleType("menace.evolution_history_db")
    hist_mod.EvolutionHistoryDB = types.SimpleNamespace
    hist_mod.EvolutionEvent = object
    sys.modules.setdefault("menace.evolution_history_db", hist_mod)
    eval_mod = types.ModuleType("menace.evaluation_history_db")
    eval_mod.EvaluationHistoryDB = types.SimpleNamespace
    sys.modules.setdefault("menace.evaluation_history_db", eval_mod)
    trend_mod = types.ModuleType("menace.trend_predictor")
    trend_mod.TrendPredictor = types.SimpleNamespace
    sys.modules.setdefault("menace.trend_predictor", trend_mod)
    mlog = types.ModuleType("menace.mutation_logger")
    mlog.log_mutation = lambda *a, **k: 1
    sys.modules.setdefault("menace.mutation_logger", mlog)

    class DummyDataBot:
        def __init__(self):
            self._callbacks = []

        def get_thresholds(self, _):
            return types.SimpleNamespace(roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0)

        reload_thresholds = get_thresholds

        def subscribe_degradation(self, cb):
            self._callbacks.append(cb)

        def roi(self, _):
            return 1.0

        def average_errors(self, _):
            return 0.0

        def average_test_failures(self, _):
            return 0.0

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            event = {
                "bot": bot,
                "delta_roi": roi - 1.0,
                "delta_errors": float(errors),
                "delta_tests_failed": float(test_failures),
                "roi_baseline": 1.0,
                "errors_baseline": 0.0,
                "tests_failed_baseline": 0.0,
            }
            for cb in list(self._callbacks):
                cb(event)
            return True

    class DummyRegistry:
        def __init__(self):
            self.graph = {}

        def register_bot(self, name):
            self.graph[name] = {"module": ""}

    class SelfCodingManager:
        def __init__(self, *, bot_name, bot_registry, data_bot, evolution_orchestrator):
            self.bot_name = bot_name
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.evolution_orchestrator = evolution_orchestrator
            self.calls = []
            self.register_bot(self.bot_name)

        def register_bot(self, name: str) -> None:
            if self.bot_registry:
                self.bot_registry.register_bot(name)
                if self.data_bot and self.evolution_orchestrator:
                    self.data_bot.subscribe_degradation(
                        lambda e: self.evolution_orchestrator.register_patch_cycle(e)
                    )

        def register_patch_cycle(self, desc, context_meta=None):
            self.calls.append((desc, context_meta))

        def should_refactor(self):
            return True

    scm_mod = types.ModuleType("menace.self_coding_manager")
    scm_mod.SelfCodingManager = SelfCodingManager
    scm_mod.HelperGenerationError = RuntimeError
    sys.modules.setdefault("menace.self_coding_manager", scm_mod)

    from menace.evolution_orchestrator import EvolutionOrchestrator

    data_bot = DummyDataBot()
    registry = DummyRegistry()
    manager = SelfCodingManager(
        bot_name="main",
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=None,
    )

    orch = EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(trend_predictor=None, energy_score=lambda *a, **k: 1.0),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        history_db=types.SimpleNamespace(add=lambda *a, **k: None),
        dataset_path=tmp_path / "roi.csv",
    )

    manager.evolution_orchestrator = orch
    data_bot._callbacks.clear()
    manager.register_bot("new_bot")
    data_bot.check_degradation("new_bot", roi=0.0, errors=5.0)

    assert manager.calls
    desc, ctx = manager.calls[0]
    assert "new_bot" in desc
    assert ctx["delta_roi"] == -1.0
    assert ctx["delta_errors"] == 5.0

