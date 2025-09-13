import inspect
from pathlib import Path


def test_degradation_triggers_patch_cycle(tmp_path):
    class BotRegistry:
        def __init__(self):
            self.graph = {}
            self.update_calls: list[str] = []

        def register_bot(self, name, **kwargs):
            self.graph.setdefault(name, {})

        def update_bot(self, name, module_path, patch_id=None, commit=None):
            self.update_calls.append(module_path)
            self.graph.setdefault(name, {})["module"] = module_path

    class DataBot:
        def __init__(self):
            self._callbacks = []

        def subscribe_degradation(self, cb):
            self._callbacks.append(cb)

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

    registry = BotRegistry()
    data_bot = DataBot()

    def self_coding_managed(*, bot_registry, data_bot):
        def decorator(cls):
            bot_registry.register_bot(cls.__name__)
            bot_registry.update_bot(cls.__name__, inspect.getfile(cls))
            cls.bot_registry = bot_registry
            cls.data_bot = data_bot
            return cls
        return decorator

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class SampleBot:
        name = "sample_bot"

        def __init__(self):
            pass

    class SelfCodingManager:
        def __init__(self, *, bot_name, bot_registry, data_bot):
            self.bot_name = bot_name
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.run_calls: list[tuple[str, str]] = []

        def run_patch(self, path: Path, desc: str, *, context_meta=None, context_builder=None):
            self.run_calls.append((str(path), desc))
            self.bot_registry.update_bot(self.bot_name, str(path))

    manager = SelfCodingManager(
        bot_name="sample_bot", bot_registry=registry, data_bot=data_bot
    )

    class EvolutionOrchestrator:
        def __init__(self, data_bot, manager):
            self.data_bot = data_bot
            self.manager = manager
            self.register_calls: list[dict] = []
            data_bot.subscribe_degradation(self.register_patch_cycle)

        def register_patch_cycle(self, event: dict):
            self.register_calls.append(event)
            self.manager.run_patch(
                tmp_path / "patched_module.py", "auto_patch", context_meta=event
            )

    orch = EvolutionOrchestrator(data_bot, manager)

    SampleBot()

    data_bot.check_degradation("sample_bot", roi=0.0, errors=2.0)

    assert orch.register_calls, "register_patch_cycle not invoked"
    assert manager.run_calls, "run_patch not invoked"
    assert registry.update_calls[-1].endswith("patched_module.py")
