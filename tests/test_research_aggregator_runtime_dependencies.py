import types

import pytest


class _DummyBroker:
    active_owner = True
    active_pipeline = None
    active_sentinel = None

    def resolve(self):
        return (None, None)

    def advertise(self, **_kwargs):
        return None


def test_runtime_dependencies_fail_fast_when_broker_absent(monkeypatch):
    import importlib
    import sys
    from pathlib import Path

    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(Path(__file__).resolve().parent.parent)]
    sys.modules["menace_sandbox"] = package

    def register(name: str, module: types.ModuleType) -> types.ModuleType:
        sys.modules[f"menace_sandbox.{name}"] = module
        sys.modules[name] = module
        return module

    dummy_settings = types.SimpleNamespace(chunk_summary_cache_dir=None)
    register(
        "sandbox_settings",
        types.SimpleNamespace(SandboxSettings=lambda: dummy_settings),
    )
    register(
        "chunk_summary_cache",
        types.SimpleNamespace(ChunkSummaryCache=lambda *_a, **_k: None),
    )
    _metric = types.SimpleNamespace(inc=lambda *_a, **_k: None, observe=lambda *_a, **_k: None)
    metrics_stub = types.SimpleNamespace(
        Gauge=lambda *_a, **_k: _metric,
        bootstrap_attempts_total=_metric,
        embedding_store_latency_seconds=_metric,
    )
    register("metrics_exporter", metrics_stub)
    embeddable_stub = types.SimpleNamespace(EmbeddableDBMixin=object)
    register("embeddable_db_mixin", embeddable_stub)
    register(
        "bot_registry",
        types.SimpleNamespace(BotRegistry=type("BotRegistry", (), {"__init__": lambda self, *a, **k: None})),
    )
    register(
        "data_bot",
        types.SimpleNamespace(
            DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}),
            persist_sc_thresholds=lambda *_a, **_k: None,
        ),
    )
    dummy_broker = _DummyBroker()
    register(
        "coding_bot_interface",
        types.SimpleNamespace(
            _BOOTSTRAP_STATE=types.SimpleNamespace(depth=0, helper_promotion_callbacks=[]),
            _looks_like_pipeline_candidate=lambda obj: obj is not None,
            _bootstrap_dependency_broker=lambda: dummy_broker,
            advertise_bootstrap_placeholder=lambda dependency_broker=None, pipeline=None, manager=None, owner=True: (
                pipeline,
                manager,
            ),
            read_bootstrap_heartbeat=lambda: False,
            get_active_bootstrap_pipeline=lambda: (None, None),
            _current_bootstrap_context=lambda: None,
            _using_bootstrap_sentinel=lambda *_a, **_k: False,
            _peek_owner_promise=lambda *_a, **_k: None,
            _GLOBAL_BOOTSTRAP_COORDINATOR=types.SimpleNamespace(peek_active=lambda: None),
            _resolve_bootstrap_wait_timeout=lambda: 0.05,
            claim_bootstrap_dependency_entry=lambda **kwargs: (
                kwargs.get("pipeline"),
                None,
                kwargs.get("manager_override"),
                False,
            ),
            prepare_pipeline_for_bootstrap=lambda *_a, **_k: None,
            self_coding_managed=lambda **_k: (lambda cls: cls),
        ),
    )
    register(
        "bootstrap_helpers",
        types.SimpleNamespace(
            bootstrap_state_snapshot=lambda: {"ready": True}, ensure_bootstrapped=lambda: None
        ),
    )
    register(
        "bootstrap_gate",
        types.SimpleNamespace(resolve_bootstrap_placeholders=lambda **_k: (None, None, dummy_broker)),
    )
    readiness = types.SimpleNamespace(
        await_ready=lambda timeout=0: True,
        describe=lambda: "stub",
    )
    register("bootstrap_readiness", types.SimpleNamespace(readiness_signal=lambda: readiness))
    register(
        "self_coding_manager",
        types.SimpleNamespace(
            SelfCodingManager=type("SelfCodingManager", (), {}),
            internalize_coding_bot=lambda *a, **k: types.SimpleNamespace(pipeline=None),
        ),
    )
    register(
        "self_coding_engine",
        types.SimpleNamespace(SelfCodingEngine=type("SelfCodingEngine", (), {"__init__": lambda self, *a, **k: None})),
    )
    register("threshold_service", types.SimpleNamespace(ThresholdService=type("ThresholdService", (), {})))
    register("code_database", types.SimpleNamespace(CodeDB=type("CodeDB", (), {})))
    register("gpt_memory", types.SimpleNamespace(GPTMemoryManager=type("GPTMemoryManager", (), {})))
    register(
        "self_coding_thresholds",
        types.SimpleNamespace(
            get_thresholds=lambda *_a, **_k: types.SimpleNamespace(
                roi_drop=0, error_increase=0, test_failure_increase=0
            )
        ),
    )
    register(
        "shared_evolution_orchestrator",
        types.SimpleNamespace(get_orchestrator=lambda *_a, **_k: None),
    )
    context_builder = type("ContextBuilder", (), {})
    vector_service = types.SimpleNamespace(EmbeddableDBMixin=object, ContextBuilder=context_builder)
    sys.modules["vector_service"] = vector_service
    sys.modules["vector_service.context_builder"] = types.SimpleNamespace(ContextBuilder=context_builder)
    register(
        "context_builder_util",
        types.SimpleNamespace(create_context_builder=lambda *_a, **_k: context_builder()),
    )
    register("snippet_compressor", types.SimpleNamespace(compress_snippets=lambda *_a, **_k: None))
    register(
        "chatgpt_enhancement_bot",
        types.SimpleNamespace(
            EnhancementDB=object,
            ChatGPTEnhancementBot=type("ChatGPTEnhancementBot", (), {}),
            Enhancement=object,
        ),
    )
    register(
        "chatgpt_prediction_bot",
        types.SimpleNamespace(ChatGPTPredictionBot=type("ChatGPTPredictionBot", (), {}), IdeaFeatures=object),
    )
    register("text_research_bot", types.SimpleNamespace(TextResearchBot=type("TextResearchBot", (), {})))
    register(
        "video_research_bot",
        types.SimpleNamespace(VideoResearchBot=type("VideoResearchBot", (), {})),
    )
    register(
        "chatgpt_research_bot",
        types.SimpleNamespace(
            ChatGPTResearchBot=type("ChatGPTResearchBot", (), {}),
            Exchange=object,
            summarise_text=lambda *_a, **_k: "",
        ),
    )
    register("database_manager", types.SimpleNamespace(get_connection=lambda *_a, **_k: None, DB_PATH="/tmp/nowhere"))
    register(
        "db_router",
        types.SimpleNamespace(
            DBRouter=type("DBRouter", (), {}),
            GLOBAL_ROUTER=None,
            init_db_router=lambda *a, **k: None,
        ),
    )

    import menace_sandbox.research_aggregator_bot as rab

    broker = _DummyBroker()

    monkeypatch.setattr(rab, "_bootstrap_placeholders", lambda: (None, None, broker))
    monkeypatch.setattr(rab, "bootstrap_state_snapshot", lambda: {"ready": True})
    monkeypatch.setattr(rab, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(rab, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(rab, "read_bootstrap_heartbeat", lambda: False)
    monkeypatch.setattr(rab, "_using_bootstrap_sentinel", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(rab, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(rab, "_resolve_pipeline_cls", lambda: object)
    monkeypatch.setattr(
        rab, "_GLOBAL_BOOTSTRAP_COORDINATOR", types.SimpleNamespace(peek_active=lambda: None)
    )
    monkeypatch.setattr(rab, "_resolve_bootstrap_wait_timeout", lambda: 0.05)
    monkeypatch.setenv("MENACE_RUNTIME_DEPENDENCY_WAIT_SECS", "0.05")

    with pytest.raises(RuntimeError) as excinfo:
        rab._ensure_runtime_dependencies()

    message = str(excinfo.value)
    assert "bootstrap" in message.lower()
    assert "broker" in message.lower()
    assert "heartbeat" in message.lower()
