import sys
import types


class DummyDB:
    def fetch(self, limit=50):
        return []


class DummyDataBot:
    def __init__(self):
        self.db = DummyDB()

    def subscribe_degradation(self, cb):
        pass


class DummyCapital:
    def energy_score(self, *a, **k):
        return 1.0


class DummyHistoryDB:
    def add(self, *a, **k):
        pass


def test_pending_patch_cycle_processed(monkeypatch):
    stub_cbi = types.ModuleType("menace.coding_bot_interface")
    stub_cbi.self_coding_managed = lambda cls: cls
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", stub_cbi)
    stub_scm = types.ModuleType("menace.self_coding_manager")
    stub_scm.SelfCodingManager = object
    stub_scm.HelperGenerationError = RuntimeError
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", stub_scm)
    stub_cap = types.ModuleType("menace.capital_management_bot")
    stub_cap.CapitalManagementBot = DummyCapital
    monkeypatch.setitem(sys.modules, "menace.capital_management_bot", stub_cap)
    stub_sem = types.ModuleType("menace.system_evolution_manager")
    stub_sem.SystemEvolutionManager = object
    monkeypatch.setitem(sys.modules, "menace.system_evolution_manager", stub_sem)
    stub_ga_clone = types.ModuleType("menace.ga_clone_manager")
    stub_ga_clone.GALearningManager = object
    monkeypatch.setitem(sys.modules, "menace.ga_clone_manager", stub_ga_clone)
    stub_ga_bot = types.ModuleType("menace.genetic_algorithm_bot")
    stub_ga_bot.GeneticAlgorithmBot = object
    stub_ga_bot.GARecord = stub_ga_bot.GAStore = object
    monkeypatch.setitem(sys.modules, "menace.genetic_algorithm_bot", stub_ga_bot)
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    import menace.data_bot as db
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "persist_sc_thresholds", lambda *a, **k: None)

    from menace.evolution_orchestrator import EvolutionOrchestrator, EvolutionTrigger

    data_bot = DummyDataBot()
    capital = DummyCapital()
    history = DummyHistoryDB()
    scm = object()
    orch = EvolutionOrchestrator(
        data_bot,
        capital,
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=scm,
        history_db=history,
        triggers=EvolutionTrigger(error_rate=1.0, roi_drop=-1.0, energy_threshold=0.0),
    )

    calls = []

    def fake_degraded(event):
        calls.append(event["bot"])

    orch._on_bot_degraded = fake_degraded  # type: ignore
    orch._pending_patch_cycle.add("dummy")
    orch.run_cycle()
    assert calls == ["dummy"]
    assert not orch._pending_patch_cycle
