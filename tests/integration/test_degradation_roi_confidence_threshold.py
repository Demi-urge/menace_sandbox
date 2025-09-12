import importlib
import types


def _setup_module(tmp_path, monkeypatch):
    scm_mod = types.ModuleType("menace.self_coding_manager")
    HelperGenerationError = type("HelperGenerationError", (Exception,), {})
    scm_mod.HelperGenerationError = HelperGenerationError
    scm_mod.SelfCodingManager = object
    import sys
    sys.modules["menace.self_coding_manager"] = scm_mod
    sc_engine_mod = types.ModuleType("menace.self_coding_engine")
    sc_engine_mod.MANAGER_CONTEXT = None
    sys.modules["menace.self_coding_engine"] = sc_engine_mod

    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")
    importlib.invalidate_caches()
    __import__("dummy_module")

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

    bus = DummyBus()

    class Manager:
        bot_name = "dummy_module"

        def __init__(self):
            self.event_bus = bus
            self.generate_called = False

        def should_refactor(self) -> bool:
            return True

        def register_patch_cycle(self, *a, **k):
            pass

        def generate_and_patch(self, *a, **k):
            self.generate_called = True

    class History:
        def __init__(self):
            self.events = []

        def add(self, event):
            self.events.append(event)

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    improver = types.SimpleNamespace()
    evolver = types.SimpleNamespace()

    return bus, Manager, History, data_bot, cap_bot, improver, evolver


def test_low_confidence_skips_patch(tmp_path, monkeypatch):
    bus, Manager, History, data_bot, cap_bot, improver, evolver = _setup_module(tmp_path, monkeypatch)

    class LowConfidencePredictor:
        def predict(self, X, horizon=1):
            current_roi = X[0][0]
            seq = [[current_roi + 1.0] for _ in range(horizon)]
            conf = [[0.1] for _ in range(horizon)]
            return seq, "", conf, None

    from menace.evolution_orchestrator import EvolutionOrchestrator

    history = History()
    manager = Manager()

    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=history,
        selfcoding_manager=manager,
        event_bus=bus,
        roi_predictor=LowConfidencePredictor(),
        roi_gain_floor=0.1,
        roi_confidence_floor=0.5,
    )

    orch._on_bot_degraded(
        {
            "bot": "dummy_module",
            "roi_baseline": 1.0,
            "delta_roi": -0.1,
            "errors_baseline": 0.0,
            "delta_errors": 0.0,
            "error_threshold": 0.0,
            "roi_threshold": -0.1,
            "test_failures": 0.0,
            "test_failure_threshold": 0.0,
        }
    )

    assert not manager.generate_called
    assert ("bot:patch_skipped", {"bot": "dummy_module", "reason": "confidence"}) in bus.events
    event = history.events[-1]
    assert event.action == "skip"
    assert event.confidence == 0.1


def test_high_confidence_triggers_patch(tmp_path, monkeypatch):
    bus, Manager, History, data_bot, cap_bot, improver, evolver = _setup_module(tmp_path, monkeypatch)

    class HighConfidencePredictor:
        def predict(self, X, horizon=1):
            current_roi = X[0][0]
            seq = [[current_roi + 1.0] for _ in range(horizon)]
            conf = [[0.9] for _ in range(horizon)]
            return seq, "", conf, None

    from menace.evolution_orchestrator import EvolutionOrchestrator

    history = History()
    manager = Manager()

    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=history,
        selfcoding_manager=manager,
        event_bus=bus,
        roi_predictor=HighConfidencePredictor(),
        roi_gain_floor=0.1,
        roi_confidence_floor=0.5,
    )

    orch._on_bot_degraded(
        {
            "bot": "dummy_module",
            "roi_baseline": 1.0,
            "delta_roi": -0.1,
            "errors_baseline": 0.0,
            "delta_errors": 0.0,
            "error_threshold": 0.0,
            "roi_threshold": -0.1,
            "test_failures": 0.0,
            "test_failure_threshold": 0.0,
        }
    )

    assert manager.generate_called
    assert ("bot:hot_swapped", {"bot": "dummy_module"}) in bus.events
    event = history.events[-1]
    assert event.action == "patch"
    assert event.confidence == 0.9
