import types
import sys


def test_degraded_bot_skips_when_thresholds_not_met(tmp_path):
    # stub self_coding_manager before importing orchestrator
    scm_mod = types.ModuleType("menace.self_coding_manager")
    HelperGenerationError = type("HelperGenerationError", (Exception,), {})
    scm_mod.HelperGenerationError = HelperGenerationError
    scm_mod.SelfCodingManager = object
    sys.modules["menace.self_coding_manager"] = scm_mod
    sc_engine_mod = types.ModuleType("menace.self_coding_engine")
    sc_engine_mod.MANAGER_CONTEXT = None
    sys.modules["menace.self_coding_engine"] = sc_engine_mod
    cbi = types.ModuleType("menace.coding_bot_interface")
    cbi.self_coding_managed = lambda *a, **k: (lambda cls: cls)
    sys.modules["menace.coding_bot_interface"] = cbi

    # Import after stubbing
    from menace.evolution_orchestrator import EvolutionOrchestrator

    events = []

    class Bus:
        def publish(self, topic, event):
            events.append((topic, event))

    class Manager:
        bot_name = "bot"

        def __init__(self) -> None:
            self.event_bus = Bus()
            self.register_called = False
            self.patch_called = False

        def should_refactor(self) -> bool:
            return False

        def register_patch_cycle(self, *a, **k):
            self.register_called = True

        def run_patch(self, *a, **k):
            self.patch_called = True
            return None

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    improver = types.SimpleNamespace()
    evolver = types.SimpleNamespace()
    manager = Manager()
    hist_stub = types.SimpleNamespace(add=lambda *a, **k: None)

    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=hist_stub,
        selfcoding_manager=manager,
    )

    mod_path = tmp_path / "mod.py"
    mod_path.write_text("pass\n")
    mod_name = "mod_for_skip"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(mod_path)
    sys.modules[mod_name] = mod
    import networkx as nx
    g = nx.DiGraph()
    g.add_node(mod_name, module=str(mod_path))
    manager.bot_registry = types.SimpleNamespace(graph=g)

    orch._on_bot_degraded({"bot": mod_name})

    assert not manager.register_called
    assert not manager.patch_called
    assert any(t == "bot:patch_skipped" for t, _ in events)
