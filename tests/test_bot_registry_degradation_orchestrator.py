from menace_sandbox.bot_registry import BotRegistry


class DummyDataBot:
    def __init__(self):
        self.callbacks = []

    def subscribe_degradation(self, cb):
        self.callbacks.append(cb)

    def check_degradation(self, bot, roi=0.0, errors=0.0, test_failures=0.0):
        event = {
            "bot": bot,
            "delta_roi": roi - 1.0,
            "delta_errors": float(errors),
            "delta_tests_failed": float(test_failures),
            "roi_baseline": 1.0,
            "errors_baseline": 0.0,
            "tests_failed_baseline": 0.0,
        }
        for cb in list(self.callbacks):
            cb(event)


class DummyManager:
    def __init__(self):
        self.calls = []
        self.evolution_orchestrator = None

    def register_patch_cycle(self, desc, context_meta=None):
        self.calls.append((desc, context_meta))


class DummyOrchestrator:
    def __init__(self, manager):
        self.events = []
        self.manager = manager

    def register_patch_cycle(self, event):
        self.events.append(event)
        desc = f"auto_patch_due_to_degradation:{event.get('bot', '')}"
        self.manager.register_patch_cycle(desc, event)


def test_degradation_flows_through_orchestrator():
    data_bot = DummyDataBot()
    manager = DummyManager()
    orchestrator = DummyOrchestrator(manager)
    manager.evolution_orchestrator = orchestrator

    registry = BotRegistry()
    registry.register_bot("sample", manager=manager, data_bot=data_bot)

    data_bot.check_degradation("sample", roi=0.0, errors=5.0)

    assert orchestrator.events, "orchestrator should receive degradation event"
    assert manager.calls, "manager.register_patch_cycle should be invoked"
    desc, ctx = manager.calls[0]
    assert "sample" in desc
    assert ctx["delta_errors"] == 5.0
