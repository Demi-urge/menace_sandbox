import types
import sys


def _install_stub_threshold_service():
    service = types.ModuleType("menace_sandbox.threshold_service")
    service.threshold_service = types.SimpleNamespace(load=lambda *_a, **_k: None)
    sys.modules.setdefault("menace_sandbox.threshold_service", service)
    sys.modules.setdefault("threshold_service", service)


_install_stub_threshold_service()

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
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    scm = types.ModuleType("menace_sandbox.self_coding_manager")
    def fake_internalize(name, engine, pipeline, *, data_bot, bot_registry, **kw):
        fake_internalize.calls.append((name, data_bot))
        return types.SimpleNamespace(evolution_orchestrator=None)

    fake_internalize.calls = []
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
    ctx_mod.record_failed_tags = lambda *a, **k: None
    ctx_mod.load_failed_tags = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", ctx_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.vector_service.context_builder", ctx_mod)

    th_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
    th_mod.get_thresholds = lambda _n: types.SimpleNamespace(
        roi_drop=-1.0, error_increase=1.0, test_failure_increase=1.0
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_thresholds", th_mod)

    def simple_internalize(self, name, *, manager=None, data_bot=None):
        scm.internalize_coding_bot(
            name,
            None,
            None,
            data_bot=data_bot,
            bot_registry=self,
        )
        node = self.graph.nodes[name]
        node.pop("pending_internalization", None)
        self._internalization_retry_attempts.pop(name, None)
        if self.event_bus:
            self.event_bus.publish("bot:internalized", {"bot": name})

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_internalize_missing_coding_bot",
        simple_internalize,
        raising=False,
    )



def test_register_bot_internalizes(monkeypatch):
    _install_stub_modules(monkeypatch)
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)
    scheduled: list[str] = []

    def _schedule(self, name: str, *, delay: float | None = None) -> None:
        self._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append(name)

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_schedule_internalization_retry",
        _schedule,
        raising=False,
    )

    reg.register_bot("FooBot", is_coding_bot=True)
    assert scheduled, "internalisation retry should be scheduled"
    for name in list(scheduled):
        reg._retry_internalization(name)
    assert ("bot:internalized", {"bot": "FooBot"}) in bus.events


def test_register_bot_internalization_retry(monkeypatch):
    _install_stub_modules(monkeypatch)
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)

    scm = sys.modules["menace_sandbox.self_coding_manager"]
    original_internalize = scm.internalize_coding_bot
    attempts = {"count": 0}

    def flaky_internalize(name, engine, pipeline, *, data_bot, bot_registry, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ModuleNotFoundError("module graph still initialising")
        return original_internalize(
            name,
            engine,
            pipeline,
            data_bot=data_bot,
            bot_registry=bot_registry,
            **kwargs,
        )

    monkeypatch.setattr(
        scm,
        "internalize_coding_bot",
        flaky_internalize,
        raising=False,
    )

    class ImmediateTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or ()
            self.kwargs = kwargs or {}
            self.daemon = False

        def start(self):
            self.function(*self.args, **self.kwargs)

        def cancel(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(bot_registry.threading, "Timer", ImmediateTimer)

    reg.register_bot("BarBot", is_coding_bot=True)

    assert attempts["count"] == 2
    node = reg.graph.nodes["BarBot"]
    assert not node.get("pending_internalization")
    assert ("bot:internalized", {"bot": "BarBot"}) in bus.events
    assert "BarBot" not in reg._internalization_retry_attempts


def test_register_bot_blocks_on_missing_dependency(monkeypatch):
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    def fail_internalize(self, name, *, manager=None, data_bot=None):
        raise ModuleNotFoundError("No module named 'totally_missing_lib'")

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_internalize_missing_coding_bot",
        fail_internalize,
        raising=False,
    )

    scheduled: list[str] = []

    def _schedule(self, name: str, *, delay: float | None = None) -> None:
        self._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append(name)

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_schedule_internalization_retry",
        _schedule,
        raising=False,
    )

    reg.register_bot("BlockedBot", is_coding_bot=True)

    for name in list(scheduled):
        reg._retry_internalization(name)

    node = reg.graph.nodes["BlockedBot"]
    assert node["internalization_blocked"]["exception"] == "ModuleNotFoundError"
    assert not node.get("pending_internalization")
    assert node["internalization_blocked"]["error"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert "totally_missing_lib" in disabled["missing_dependencies"]
    assert disabled["reason"].startswith("self-coding disabled after unrecoverable import failure")
    assert any(evt[0] == "bot:internalization_blocked" for evt in bus.events)
    assert any(evt[0] == "bot:self_coding_disabled" for evt in bus.events)


def test_dependency_failure_disables_self_coding(monkeypatch):
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_thresholds",
        lambda _name: types.SimpleNamespace(
            roi_drop=None, error_increase=None, test_failure_increase=None
        ),
    )

    class _StubDataBot(DummyDataBot):
        def schedule_monitoring(self, *_a, **_k):
            raise AssertionError("should not schedule monitoring when disabled")

    components = bot_registry._SelfCodingComponents(
        internalize_coding_bot=lambda *a, **k: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'torch'")
        ),
        engine_cls=lambda *a, **k: object(),
        pipeline_cls=lambda *a, **k: object(),
        data_bot_cls=lambda *a, **k: _StubDataBot(),
        code_db_cls=lambda *a, **k: object(),
        memory_manager_cls=lambda *a, **k: object(),
        context_builder_factory=lambda: object(),
    )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_components",
        lambda: components,
    )

    scheduled: list[str] = []

    def _schedule(self, name: str, *, delay: float | None = None) -> None:
        self._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append(name)

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_schedule_internalization_retry",
        _schedule,
        raising=False,
    )

    reg.register_bot("TorchlessBot", is_coding_bot=True)

    for name in list(scheduled):
        reg._retry_internalization(name)

    node = reg.graph.nodes["TorchlessBot"]
    disabled = node["self_coding_disabled"]
    assert "torch" in disabled["missing_dependencies"]
    assert disabled["reason"]
    assert any(evt[0] == "bot:self_coding_disabled" for evt in bus.events)


def test_dependency_probe_used_when_import_error_ambiguous(monkeypatch):
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_thresholds",
        lambda _name: types.SimpleNamespace(
            roi_drop=None, error_increase=None, test_failure_increase=None
        ),
    )

    components = bot_registry._SelfCodingComponents(
        internalize_coding_bot=lambda *a, **k: (_ for _ in ()).throw(
            ModuleNotFoundError("DLL load failed: error 193")
        ),
        engine_cls=lambda *a, **k: object(),
        pipeline_cls=lambda *a, **k: object(),
        data_bot_cls=lambda *a, **k: DummyDataBot(),
        code_db_cls=lambda *a, **k: object(),
        memory_manager_cls=lambda *a, **k: object(),
        context_builder_factory=lambda: DummyContext(),
    )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_components",
        lambda: components,
    )

    monkeypatch.setattr(
        bot_registry,
        "ensure_self_coding_ready",
        lambda: (False, ("helper_lib",)),
    )

    reg.register_bot("AmbiguousBot", is_coding_bot=True)

    node = reg.graph.nodes["AmbiguousBot"]
    disabled = node["self_coding_disabled"]
    assert disabled["missing_dependencies"] == ["helper_lib"]
    assert any(evt[0] == "bot:self_coding_disabled" for evt in bus.events)


def test_import_error_without_hints_disables_self_coding(monkeypatch):
    bus = DummyBus()
    reg = bot_registry.BotRegistry(event_bus=bus)

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_thresholds",
        lambda _name: types.SimpleNamespace(
            roi_drop=None, error_increase=None, test_failure_increase=None
        ),
    )

    components = bot_registry._SelfCodingComponents(
        internalize_coding_bot=lambda *a, **k: (_ for _ in ()).throw(
            ImportError("loader returned NULL without setting an error")
        ),
        engine_cls=lambda *a, **k: object(),
        pipeline_cls=lambda *a, **k: object(),
        data_bot_cls=lambda *a, **k: DummyDataBot(),
        code_db_cls=lambda *a, **k: object(),
        memory_manager_cls=lambda *a, **k: object(),
        context_builder_factory=lambda: DummyContext(),
    )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_components",
        lambda: components,
    )

    monkeypatch.setattr(
        bot_registry,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )

    scheduled: list[str] = []

    def _schedule(self, name: str, *, delay: float | None = None) -> None:
        self._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append(name)

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "_schedule_internalization_retry",
        _schedule,
        raising=False,
    )

    reg.register_bot("NullImportBot", is_coding_bot=True)

    for name in list(scheduled):
        reg._retry_internalization(name)

    node = reg.graph.nodes["NullImportBot"]
    disabled = node["self_coding_disabled"]
    assert "self_coding_runtime" in disabled["missing_dependencies"]
    assert "unresolved import error" in disabled["reason"]
    assert any(evt[0] == "bot:self_coding_disabled" for evt in bus.events)
