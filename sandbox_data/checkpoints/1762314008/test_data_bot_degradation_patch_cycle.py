import types
import sys
from pathlib import Path


class DummyBus:
    def __init__(self):
        self.subs = {}
        self.events = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.events.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummySelfCodingManager:
    def __init__(self, module: Path):
        self.bot_name = "dummy"
        self.bot_registry = types.SimpleNamespace(graph={"dummy": {"module": str(module)}})
        self.calls: list[tuple[str, dict | None]] = []

    def register_patch_cycle(self, desc: str, ctx=None, **_k):
        self.calls.append((desc, ctx))


class DummyCapital:
    trend_predictor = object()

    def energy_score(self, *a, **k):  # pragma: no cover - constant
        return 1.0


class DummyHistoryDB:
    def add(self, *a, **k):  # pragma: no cover - noop
        pass


def test_threshold_breach_queues_patch_cycle(tmp_path, monkeypatch):
    stub_scm = types.ModuleType("menace.self_coding_manager")
    stub_scm.SelfCodingManager = DummySelfCodingManager
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
    stub_mut = types.ModuleType("menace.mutation_logger")
    stub_mut.log_mutation = lambda *a, **k: None
    stub_mut.record_mutation_outcome = lambda *a, **k: None
    class _Ctx:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    stub_mut.log_context = _Ctx
    monkeypatch.setitem(sys.modules, "menace.mutation_logger", stub_mut)
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    import menace.data_bot as db
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "persist_sc_thresholds", lambda *a, **k: None)

    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )

    bus = DummyBus()
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(
        mdb,
        settings=settings,
        event_bus=bus,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
    )

    mod = tmp_path / "dummy.py"
    mod.write_text("def foo():\n    return 1\n")
    manager = DummySelfCodingManager(mod)

    from menace.evolution_orchestrator import EvolutionOrchestrator

    orch = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
        roi_gain_floor=1.0,  # force decision skip before patching
    )

    orch.register_bot("dummy")
    data_bot.check_degradation("dummy", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy", roi=0.0, errors=5.0)

    assert manager.calls, "patch cycle should be registered"
    assert "dummy" in orch._pending_patch_cycle
    assert any(t == "bot:degraded" for t, _ in bus.events)
