import sys
import types
from pathlib import Path


def test_threshold_reload_propagates(tmp_path, monkeypatch):
    ROOT = Path(__file__).resolve().parents[2]
    package = types.ModuleType("menace")
    package.__path__ = [str(ROOT)]
    sys.modules.setdefault("menace", package)

    # temporary threshold config
    cfg = tmp_path / "self_coding_thresholds.yaml"
    cfg.write_text(
        """default:\n  roi_drop: -0.1\n  error_increase: 1.0\n  test_failure_increase: 0.0\nbots:\n  dummy:\n    roi_drop: -0.1\n    error_increase: 1.0\n    test_failure_increase: 0.0\n""",
        encoding="utf-8",
    )
    import menace.self_coding_thresholds as sct
    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg)

    class DummyBus:
        def __init__(self):
            self.subs = {}
        def subscribe(self, topic, fn):
            self.subs.setdefault(topic, []).append(fn)
        def publish(self, topic, payload):
            for fn in self.subs.get(topic, []):
                fn(topic, payload)

    bus = DummyBus()

    from menace.data_bot import DataBot, MetricsDB
    db = MetricsDB(path=tmp_path / "metrics.db")
    data_bot = DataBot(db=db, event_bus=bus)

    cbi = types.ModuleType("menace.coding_bot_interface")
    cbi.self_coding_managed = lambda f: f
    cbi.manager_generate_helper = lambda *a, **k: ""
    sys.modules["menace.coding_bot_interface"] = cbi
    sce = types.ModuleType("menace.self_coding_engine")
    sce.MANAGER_CONTEXT = None
    sys.modules["menace.self_coding_engine"] = sce
    sem = types.ModuleType("menace.system_evolution_manager")
    sem.SystemEvolutionManager = object
    sys.modules["menace.system_evolution_manager"] = sem
    ehdb = types.ModuleType("menace.evolution_history_db")
    ehdb.EvolutionHistoryDB = object
    ehdb.EvolutionEvent = object
    sys.modules["menace.evolution_history_db"] = ehdb
    evh = types.ModuleType("menace.evaluation_history_db")
    evh.EvaluationHistoryDB = object
    sys.modules["menace.evaluation_history_db"] = evh
    tp = types.ModuleType("menace.trend_predictor")
    tp.TrendPredictor = object
    sys.modules["menace.trend_predictor"] = tp
    cmb = types.ModuleType("menace.capital_management_bot")
    cmb.CapitalManagementBot = object
    sys.modules["menace.capital_management_bot"] = cmb
    scm_stub = types.ModuleType("menace.self_coding_manager")
    scm_stub.HelperGenerationError = RuntimeError
    sys.modules["menace.self_coding_manager"] = scm_stub

    class SelfCodingManager:
        def __init__(self, data_bot, bot_name, event_bus):
            self.data_bot = data_bot
            self.bot_name = bot_name
            self.event_bus = event_bus
            self.roi_drop_threshold = 0.0
            self.error_rate_threshold = 0.0
            self.test_failure_threshold = 0.0
            self._last_thresholds = None
            self._refresh_thresholds()

        def _refresh_thresholds(self):
            prev = self._last_thresholds
            t = self.data_bot.reload_thresholds(self.bot_name)
            self.roi_drop_threshold = t.roi_drop
            self.error_rate_threshold = t.error_threshold
            self.test_failure_threshold = t.test_failure_threshold
            self._last_thresholds = t
            if prev != t and self.event_bus:
                payload = {
                    "bot": self.bot_name,
                    "roi_drop": t.roi_drop,
                    "error_threshold": t.error_threshold,
                    "test_failure_threshold": t.test_failure_threshold,
                }
                self.event_bus.publish("self_coding:thresholds_updated", payload)

    manager = SelfCodingManager(data_bot, "dummy", bus)

    from menace.evolution_orchestrator import EvolutionOrchestrator
    orch = EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(trend_predictor=None),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=types.SimpleNamespace(add=lambda *a, **k: None),
    )

    assert orch.triggers.error_rate == 1.0
    assert orch.triggers.roi_drop == -0.1

    cfg.write_text(
        """default:\n  roi_drop: -0.1\n  error_increase: 1.0\n  test_failure_increase: 0.0\nbots:\n  dummy:\n    roi_drop: -0.2\n    error_increase: 2.0\n    test_failure_increase: 0.0\n""",
        encoding="utf-8",
    )

    manager._refresh_thresholds()

    assert orch.triggers.error_rate == 2.0
    assert orch.triggers.roi_drop == -0.2
