import importlib
import types
import sys
import os
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

pkg_path = Path(__file__).resolve().parent.parent
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(pkg_path)]
sys.modules.setdefault("menace_sandbox", pkg)

enh_module = importlib.import_module("menace_sandbox.enhancement_bot")
EnhancementBot = enh_module.EnhancementBot
RefactorProposal = enh_module.RefactorProposal

UnifiedEventBus = importlib.import_module("menace_sandbox.unified_event_bus").UnifiedEventBus
BotRegistry = importlib.import_module("menace_sandbox.bot_registry").BotRegistry


class DummyManager:
    def __init__(self):
        self.event_bus = UnifiedEventBus()
        self.bot_registry = BotRegistry(event_bus=self.event_bus)
        self.patches = []

    def register_bot(self, name: str) -> None:
        self.bot_registry.register_bot(name)

    def run_patch(self, path: Path, desc: str, **kwargs):
        self.patches.append((path, desc))
        self.event_bus.publish("self_coding:patch_applied", {"module": str(path)})


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


class DummyLLM:
    def generate(self, prompt, context_builder=None):
        return types.SimpleNamespace(text="summary")


def test_enhancement_bot_emits_event_and_registry(tmp_path):
    manager = DummyManager()
    events = []
    manager.event_bus.subscribe("self_coding:patch_applied", lambda t, e: events.append(e))

    class DummyDB:
        def record_history(self, *a, **k):
            pass
        def add(self, *a, **k):
            pass

    bot = EnhancementBot(code_db=object(), enhancement_db=DummyDB(), context_builder=DummyContextBuilder(), llm_client=DummyLLM(), manager=manager)
    bot._logic_preserved = lambda a, b: True
    bot._benchmark = lambda code: 1.0
    bot._codex_summarize = lambda *a, **k: ""
    file_path = tmp_path / "mod.py"
    file_path.write_text("def run():\n    return 1\n")
    proposal = RefactorProposal(file_path=file_path, new_code="def run():\n    return 2\n")
    assert bot.evaluate(proposal)
    assert events, "patch event not emitted"
    name = bot.__class__.__name__
    assert manager.bot_registry.graph.nodes[name]["module"] == str(file_path)
