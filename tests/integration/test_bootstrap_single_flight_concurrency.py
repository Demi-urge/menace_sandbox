import logging
import threading
from types import SimpleNamespace

import coding_bot_interface as cbi

try:
    import menace_sandbox.research_aggregator_bot as rab  # type: ignore
except ImportError:  # pragma: no cover - flat layout support
    import research_aggregator_bot as rab  # type: ignore

try:
    import menace_sandbox.watchdog as watchdog_module  # type: ignore
except ImportError:  # pragma: no cover - flat layout support
    import watchdog as watchdog_module  # type: ignore


def test_late_consumers_reuse_placeholder_during_single_flight(monkeypatch):
    class _Broker:
        def __init__(self) -> None:
            self.pipeline: object | None = None
            self.sentinel: object | None = None
            self.active_owner = False

        def resolve(self) -> tuple[object | None, object | None]:
            return self.pipeline, self.sentinel

        def advertise(
            self, *, pipeline: object | None = None, sentinel: object | None = None, owner: bool = False
        ) -> tuple[object | None, object | None]:
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            self.active_owner = owner or self.active_owner
            return self.pipeline, self.sentinel

        def clear(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.active_owner = False

    broker = _Broker()
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()

    for attr in (
        "depth",
        "sentinel_manager",
        "pipeline",
        "owner_depths",
        "active_bootstrap_guard",
        "active_bootstrap_token",
    ):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    start_event = threading.Event()
    release_event = threading.Event()
    prepare_calls: list[object] = []
    promotions: list[object] = []

    def _stub_inner(**_kwargs):
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
        start_event.set()
        release_event.wait(timeout=5)
        return pipeline_placeholder, lambda manager: promotions.append(manager)

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    real_prepare = cbi.prepare_pipeline_for_bootstrap

    def _spy_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return real_prepare(**kwargs)

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _spy_prepare)
    monkeypatch.setattr(rab, "prepare_pipeline_for_bootstrap", _spy_prepare)
    monkeypatch.setattr(watchdog_module, "prepare_pipeline_for_bootstrap", _spy_prepare)

    def _start_component_timers(self, *args, **kwargs):
        start_event.set()
        return {}

    monkeypatch.setattr(
        cbi.SharedTimeoutCoordinator, "start_component_timers", _start_component_timers
    )

    class _Pipeline:
        vector_bootstrap_heavy = True

        def __init__(self, *, manager: object, **_kwargs) -> None:
            self.manager = manager

    owner_thread = threading.Thread(
        target=cbi.prepare_pipeline_for_bootstrap,
        kwargs={
            "pipeline_cls": _Pipeline,
            "context_builder": SimpleNamespace(),
            "bot_registry": SimpleNamespace(),
            "data_bot": SimpleNamespace(),
        },
    )
    owner_thread.start()
    assert start_event.wait(timeout=5)

    rab.registry = None
    rab.data_bot = None
    rab._context_builder = None
    rab.engine = None
    rab._PipelineCls = None
    rab.pipeline = None
    rab.evolution_orchestrator = None
    rab.manager = None
    rab._runtime_state = None
    rab._runtime_placeholder = None
    rab._runtime_initializing = False
    rab._self_coding_configured = False

    class _Registry:
        pass

    class _DataBot:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    monkeypatch.setattr(rab, "BotRegistry", _Registry)
    monkeypatch.setattr(rab, "DataBot", _DataBot)
    monkeypatch.setattr(rab, "create_context_builder", lambda: _ContextBuilder())
    monkeypatch.setattr(rab, "SelfCodingEngine", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "CodeDB", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "GPTMemoryManager", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "get_orchestrator", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        rab,
        "internalize_coding_bot",
        lambda *args, **kwargs: SimpleNamespace(pipeline=pipeline_placeholder),
    )
    monkeypatch.setattr(
        rab,
        "self_coding_managed",
        lambda **_: (lambda cls: cls),
    )

    aggregator_state = rab._ensure_runtime_dependencies(
        promote_pipeline=lambda manager: promotions.append(manager),
    )

    monkeypatch.setattr(watchdog_module, "UnifiedEventBus", None)
    monkeypatch.setattr(watchdog_module, "SelfCodingEngine", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "CodeDB", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "MenaceMemoryManager", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        watchdog_module,
        "internalize_coding_bot",
        lambda *args, **kwargs: SimpleNamespace(
            evolution_orchestrator="eo", quick_fix="qf", pipeline=pipeline_placeholder
        ),
    )
    monkeypatch.setattr(watchdog_module, "persist_sc_thresholds", lambda *args, **kwargs: None)
    monkeypatch.setattr(watchdog_module, "DATA_BOT", SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "REGISTRY", SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "ModelAutomationPipeline", _Pipeline)
    monkeypatch.setattr(watchdog_module, "SelfHealingOrchestrator", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "KnowledgeGraph", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(watchdog_module, "EscalationProtocol", lambda levels: SimpleNamespace(levels=levels))
    monkeypatch.setattr(watchdog_module, "EscalationLevel", lambda name, notifier: (name, notifier))
    monkeypatch.setattr(
        watchdog_module, "Thresholds", lambda: SimpleNamespace(roi_loss_percent=1.0, error_trend=2.0)
    )
    monkeypatch.setattr(watchdog_module, "Notifier", lambda: SimpleNamespace(auto_handler=None))
    monkeypatch.setattr(watchdog_module, "_default_auto_handler", lambda *_: lambda **__: None)

    watchdog_instance = watchdog_module.Watchdog(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        context_builder=SimpleNamespace(),
        registry=None,
    )

    release_event.set()
    owner_thread.join(timeout=5)

    assert len(prepare_calls) == 1
    assert aggregator_state.pipeline is pipeline_placeholder
    assert watchdog_instance.manager.pipeline is pipeline_placeholder
    assert promotions


def test_multi_module_bootstrap_logs_once(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    class _Broker:
        def __init__(self) -> None:
            self.pipeline: object | None = None
            self.sentinel: object | None = None
            self.active_owner = False

        def resolve(self) -> tuple[object | None, object | None]:
            return self.pipeline, self.sentinel

        def advertise(
            self, *, pipeline: object | None = None, sentinel: object | None = None, owner: bool = False
        ) -> tuple[object | None, object | None]:
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            self.active_owner = owner or self.active_owner
            return self.pipeline, self.sentinel

        def clear(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.active_owner = False

    broker = _Broker()
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()
    cbi._REENTRY_ATTEMPTS.clear()

    for attr in (
        "depth",
        "sentinel_manager",
        "pipeline",
        "owner_depths",
        "active_bootstrap_guard",
        "active_bootstrap_token",
    ):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)

    start_event = threading.Event()
    release_event = threading.Event()
    prepare_calls: list[object] = []
    promotions: list[object | None] = []

    def _stub_inner(**_kwargs):
        prepare_calls.append(_kwargs)
        start_event.set()
        release_event.wait(timeout=5)
        return SimpleNamespace(manager="manager"), promotions.append

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    class _Pipeline:
        def __init__(self, *, manager: object, **_kwargs) -> None:
            self.manager = manager

    barrier = threading.Barrier(3)
    results: list[tuple[object, object]] = []

    def _bootstrap_from(module_name: str) -> None:
        barrier.wait()
        pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=_Pipeline,
            context_builder=SimpleNamespace(),
            bot_registry=SimpleNamespace(),
            data_bot=SimpleNamespace(),
            helper=module_name,
        )
        promote(pipeline.manager)
        results.append((module_name, pipeline))

    threads = [
        threading.Thread(target=_bootstrap_from, args=(name,))
        for name in (
            "menace_sandbox.research_aggregator_bot",
            "menace_sandbox.watchdog",
            "menace_sandbox.prediction_manager_bot",
        )
    ]

    for thread in threads:
        thread.start()

    assert start_event.wait(timeout=5)
    release_event.set()

    for thread in threads:
        thread.join(timeout=5)

    assert len(prepare_calls) == 1, "bootstrap implementation should run only once"
    assert len(promotions) == len(threads), "each caller should reuse the single-flight pipeline"
    assert all(result[1] is results[0][1] for result in results)

    prepare_logs = [
        record for record in caplog.records if "prepare_pipeline_for_bootstrap" in record.getMessage()
    ]
    assert len(prepare_logs) == 1, "prepare_pipeline_for_bootstrap should log once per process"
    assert not any("recursion_guard" in record.getMessage() for record in caplog.records)

