import types
import sys

from menace_sandbox import bot_registry

class DummyBus:
    def __init__(self):
        self.events = []
    def publish(self, topic, payload):
        self.events.append((topic, payload))

class DummyDataBot:
    def __init__(self, *a, **k):
        pass
    def check_degradation(self, *a, **k):
        pass
    def subscribe_degradation(self, *a, **k):
        pass

class Dummy:
    def __init__(self, *a, **k):
        pass

class DummyContext:
    def refresh_db_weights(self):
        pass


def _install_stub_modules(monkeypatch):
    scm = types.ModuleType("menace_sandbox.self_coding_manager")
    def fake_internalize(name, engine, pipeline, *, data_bot, bot_registry, **kw):
        mgr = types.SimpleNamespace(evolution_orchestrator=None)
        bot_registry.register_bot(name, manager=mgr, data_bot=data_bot, is_coding_bot=True)
        return mgr
    scm.internalize_coding_bot = fake_internalize
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", scm)

    db_mod = types.ModuleType("menace_sandbox.data_bot")
    db_mod.DataBot = DummyDataBot
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", db_mod)

    eng_mod = types.ModuleType("menace_sandbox.self_coding_engine")
    eng_mod.SelfCodingEngine = Dummy
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", eng_mod)

    pipe_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")
    pipe_mod.ModelAutomationPipeline = Dummy
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", pipe_mod)

    code_mod = types.ModuleType("menace_sandbox.code_database")
    code_mod.CodeDB = Dummy
    monkeypatch.setitem(sys.modules, "menace_sandbox.code_database", code_mod)

    mem_mod = types.ModuleType("menace_sandbox.gpt_memory")
    mem_mod.GPTMemoryManager = Dummy
    monkeypatch.setitem(sys.modules, "menace_sandbox.gpt_memory", mem_mod)

    ctx_mod = types.ModuleType("vector_service.context_builder")
    ctx_mod.ContextBuilder = DummyContext
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", ctx_mod)

    th_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
    th_mod.get_thresholds = lambda _n: types.SimpleNamespace(
        roi_drop=-1.0, error_increase=1.0, test_failure_increase=1.0
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_thresholds", th_mod)


def test_register_bot_internalizes(monkeypatch):
    _install_stub_modules(monkeypatch)
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)
    reg.register_bot("FooBot", is_coding_bot=True)
    assert ("bot:internalized", {"bot": "FooBot"}) in bus.events
