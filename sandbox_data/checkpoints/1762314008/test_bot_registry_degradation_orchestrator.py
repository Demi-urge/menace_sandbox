from pathlib import Path
import types

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


class DummyEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, payload):
        self.events.append((topic, payload))


class DummySelfTestManager:
    def __init__(self, summary=None, *, fail=False, error_message="validation failed"):
        self.summary = summary or {"self_tests": {"passed": 1}}
        self.fail = fail
        self.error_message = error_message
        self.gen_calls = []
        self.post_calls = []
        self.register_calls = []
        self.refresh_calls = 0
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="token-123")

    def register_patch_cycle(self, desc, context_meta=None, provenance_token=None):
        self.register_calls.append(
            {
                "description": desc,
                "context_meta": context_meta,
                "provenance_token": provenance_token,
            }
        )
        return 42, None

    def refresh_quick_fix_context(self):
        self.refresh_calls += 1
        return object()

    def generate_and_patch(
        self,
        path,
        description,
        *,
        context_meta=None,
        context_builder=None,
        provenance_token=None,
    ):
        self.gen_calls.append(
            {
                "path": Path(path),
                "description": description,
                "context_meta": dict(context_meta or {}),
                "context_builder": context_builder,
                "provenance_token": provenance_token,
            }
        )
        return types.SimpleNamespace(status="ok"), "new-commit"

    def run_post_patch_cycle(
        self,
        module_path,
        description,
        *,
        provenance_token,
        context_meta=None,
    ):
        call = {
            "module_path": Path(module_path),
            "description": description,
            "provenance_token": provenance_token,
            "context_meta": dict(context_meta or {}),
        }
        self.post_calls.append(call)
        if self.fail:
            raise RuntimeError(self.error_message)
        return self.summary


def test_degradation_flows_through_orchestrator():
    data_bot = DummyDataBot()
    manager = DummyManager()
    orchestrator = DummyOrchestrator(manager)
    manager.evolution_orchestrator = orchestrator

    registry = BotRegistry()
    registry.register_bot("sample", manager=manager, data_bot=data_bot, is_coding_bot=True)

    data_bot.check_degradation("sample", roi=0.0, errors=5.0)

    assert orchestrator.events, "orchestrator should receive degradation event"
    assert manager.calls, "manager.register_patch_cycle should be invoked"
    desc, ctx = manager.calls[0]
    assert "sample" in desc
    assert ctx["delta_errors"] == 5.0


def test_registry_triggers_post_patch_self_test(tmp_path):
    data_bot = DummyDataBot()
    manager = DummySelfTestManager(summary={"self_tests": {"passed": 2, "failed": 0}})
    bus = DummyEventBus()
    registry = BotRegistry(event_bus=bus)
    registry.register_bot("sample", manager=manager, data_bot=data_bot, is_coding_bot=True)

    module_path = tmp_path / "bot_module.py"
    module_path.write_text("print('hi')")
    registry.graph.nodes["sample"]["module"] = str(module_path)

    data_bot.check_degradation("sample", roi=0.0, errors=3.0)

    assert manager.post_calls, "post patch cycle not invoked"
    post_call = manager.post_calls[0]
    assert post_call["module_path"] == module_path
    assert post_call["description"].startswith("auto_patch_due_to_degradation:sample")
    assert post_call["context_meta"]["delta_errors"] == 3.0
    assert post_call["provenance_token"] == "token-123"

    bot_event = next(p for t, p in bus.events if t == "bot:patch_applied")
    assert bot_event["post_validation"] == {"self_tests": {"passed": 2, "failed": 0}}
    assert bot_event["commit"] == "new-commit"


def test_registry_surfaces_post_patch_validation_error(tmp_path):
    data_bot = DummyDataBot()
    manager = DummySelfTestManager(fail=True, error_message="self test failure")
    bus = DummyEventBus()
    registry = BotRegistry(event_bus=bus)
    registry.register_bot("sample", manager=manager, data_bot=data_bot, is_coding_bot=True)

    module_path = tmp_path / "bot_module.py"
    module_path.write_text("print('hi')")
    registry.graph.nodes["sample"]["module"] = str(module_path)

    data_bot.check_degradation("sample", roi=0.0, errors=4.0)

    assert manager.post_calls, "post patch cycle not attempted"
    failure_event = next(p for t, p in bus.events if t == "bot:patch_failed")
    assert failure_event["post_validation_error"] == "self test failure"
    assert failure_event["commit"] == "new-commit"
    assert not any(t == "bot:patch_applied" for t, _ in bus.events)
