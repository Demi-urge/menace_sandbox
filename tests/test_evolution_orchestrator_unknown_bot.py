import types
import sys


def test_unknown_bot_emits_event(tmp_path):
    scm_mod = types.ModuleType("menace.self_coding_manager")
    HelperGenerationError = type("HelperGenerationError", (Exception,), {})
    scm_mod.HelperGenerationError = HelperGenerationError
    scm_mod.SelfCodingManager = object
    sys.modules["menace.self_coding_manager"] = scm_mod
    sc_engine_mod = types.ModuleType("menace.self_coding_engine")
    sc_engine_mod.MANAGER_CONTEXT = None
    sys.modules["menace.self_coding_engine"] = sc_engine_mod

    from menace.evolution_orchestrator import EvolutionOrchestrator

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload):
            events.append((topic, payload))

    manager = types.SimpleNamespace(bot_name="bot", event_bus=Bus())
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    improver = types.SimpleNamespace()
    evolver = types.SimpleNamespace()

    history = types.SimpleNamespace(add=lambda *a, **k: None)
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=history,
        selfcoding_manager=manager,
    )

    orch._on_bot_degraded({"bot": "missing_bot"})

    assert ("evolve:unknown_bot", {"bot": "missing_bot"}) in events

