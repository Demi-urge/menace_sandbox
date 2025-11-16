import sys
import types


def test_metric_drop_triggers_patch_cycle(monkeypatch):
    # stub vector_service with minimal components
    vs = types.ModuleType("vector_service")

    class ContextBuilder:
        def build(self, prompt, session_id=None, include_vectors=False, **_):
            return {}

        def refresh_db_weights(self):
            pass

    class CognitionLayer:
        def __init__(self, *, context_builder=None, **_):
            self.context_builder = context_builder

    class FallbackResult:
        pass

    class ErrorResult(Exception):
        pass

    vs.ContextBuilder = ContextBuilder
    vs.CognitionLayer = CognitionLayer
    vs.FallbackResult = FallbackResult
    vs.ErrorResult = ErrorResult
    monkeypatch.setitem(sys.modules, "vector_service", vs)

    # lightweight ensure_fresh_weights
    cbu = types.ModuleType("context_builder_util")
    cbu.ensure_fresh_weights = lambda b: b.refresh_db_weights()
    monkeypatch.setitem(sys.modules, "context_builder_util", cbu)

    # self_coding_managed decorator stub
    cbi = types.ModuleType("coding_bot_interface")
    cbi.self_coding_managed = lambda cls=None, *a, **k: (cls if cls else (lambda c: c))
    monkeypatch.setitem(sys.modules, "coding_bot_interface", cbi)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi)

    # minimal unified_event_bus to avoid circular imports
    bus_mod = types.ModuleType("unified_event_bus")

    class UnifiedEventBus:
        def subscribe(self, *a, **k):
            pass

        def publish(self, *a, **k):
            pass

    bus_mod.UnifiedEventBus = UnifiedEventBus
    monkeypatch.setitem(sys.modules, "unified_event_bus", bus_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.unified_event_bus", bus_mod)

    # DataBot stub that immediately signals degradation
    db_mod = types.ModuleType("data_bot")

    class DummyDataBot:
        def __init__(self, *a, event_bus=None, **k):
            self.callbacks = []

        def roi(self, bot):
            return 0.0

        def average_errors(self, bot):
            return 0.0

        def record_metrics(self, bot, roi, errors, test_failures=0.0):
            pass

        def subscribe_degradation(self, cb):
            self.callbacks.append(cb)

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            event = {
                "bot": bot,
                "roi_baseline": 1.0,
                "delta_roi": roi - 1.0,
                "delta_errors": errors,
                "errors_baseline": 0.0,
            }
            for cb in self.callbacks:
                cb(event)
            return True

    db_mod.DataBot = DummyDataBot
    monkeypatch.setitem(sys.modules, "data_bot", db_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", db_mod)

    import menace_sandbox.automated_reviewer as ar

    class DummyEscalationManager:
        def handle(self, *a, **k):
            pass

    class DummyBotDB:
        def update_bot(self, *a, **k):
            pass

    data_bot = ar.data_bot

    class DummyOrchestrator:
        def __init__(self, data_bot, manager):
            self.data_bot = data_bot
            self.manager = manager
            self.registered = []
            self.subscribed = False

        def register_bot(self, bot):
            self.registered.append(bot)

        def _ensure_degradation_subscription(self):
            if self.subscribed or not self.registered:
                return
            self.data_bot.subscribe_degradation(self._on_degraded)
            self.subscribed = True

        def _on_degraded(self, event):
            desc = f"auto_patch_due_to_degradation:{event.get('bot')}"
            self.manager.register_patch_cycle(desc, event)

    class DummyManager:
        def __init__(self, bot_name, data_bot):
            self.bot_name = bot_name
            self.manager_generate_helper = lambda *a, **k: None
            self.register_patch_cycle_calls = []
            self.evolution_orchestrator = DummyOrchestrator(data_bot, self)

        def register_patch_cycle(self, desc, context_meta=None):
            self.register_patch_cycle_calls.append((desc, context_meta))

    manager = DummyManager("123", data_bot)
    builder = vs.ContextBuilder()
    reviewer = ar.AutomatedReviewer(
        builder,
        manager=manager,
        bot_db=DummyBotDB(),
        escalation_manager=DummyEscalationManager(),
    )

    reviewer.handle({"bot_id": "123", "severity": "critical"})

    assert manager.register_patch_cycle_calls
