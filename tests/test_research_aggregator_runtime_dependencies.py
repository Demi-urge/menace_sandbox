import logging
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


def _setup_runtime_dependencies(monkeypatch, *, heartbeat: bool = False, max_wait: float | None = None):
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
    broker = _DummyBroker()
    register(
        "coding_bot_interface",
        types.SimpleNamespace(
            _BOOTSTRAP_STATE=types.SimpleNamespace(depth=0, helper_promotion_callbacks=[]),
            _looks_like_pipeline_candidate=lambda obj: obj is not None,
            _bootstrap_dependency_broker=lambda: broker,
            advertise_bootstrap_placeholder=lambda dependency_broker=None, pipeline=None, manager=None, owner=True: (
                pipeline,
                manager,
            ),
            read_bootstrap_heartbeat=lambda: heartbeat,
            get_active_bootstrap_pipeline=lambda: (None, None),
            _current_bootstrap_context=lambda: None,
            _using_bootstrap_sentinel=lambda *_a, **_k: False,
            _peek_owner_promise=lambda *_a, **_k: None,
            _GLOBAL_BOOTSTRAP_COORDINATOR=types.SimpleNamespace(peek_active=lambda: None),
            _resolve_bootstrap_wait_timeout=lambda **_k: 0.05,
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
        types.SimpleNamespace(resolve_bootstrap_placeholders=lambda **_k: (None, None, broker)),
    )
    readiness = types.SimpleNamespace(
        await_ready=lambda timeout=0: True,
        describe=lambda: "stub",
    )
    register(
        "bootstrap_readiness",
        types.SimpleNamespace(
            readiness_signal=lambda: readiness,
            probe_embedding_service=lambda *_a, **_k: None,
        ),
    )
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
        )
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

    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")

    monkeypatch.setattr(rab, "_bootstrap_placeholders", lambda **_k: (None, None, broker))
    monkeypatch.setattr(rab, "bootstrap_state_snapshot", lambda: {"ready": True})
    monkeypatch.setattr(rab, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(rab, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(rab, "_using_bootstrap_sentinel", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(rab, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(rab, "_resolve_pipeline_cls", lambda: object)
    monkeypatch.setattr(
        rab, "_GLOBAL_BOOTSTRAP_COORDINATOR", types.SimpleNamespace(peek_active=lambda: None)
    )
    monkeypatch.setattr(rab, "_resolve_bootstrap_wait_timeout", lambda: 0.05)
    monkeypatch.setenv("MENACE_RUNTIME_DEPENDENCY_WAIT_SECS", "0.05")
    if max_wait is not None:
        monkeypatch.setenv("MENACE_RUNTIME_DEPENDENCY_MAX_WAIT_SECS", str(max_wait))
    else:
        monkeypatch.delenv("MENACE_RUNTIME_DEPENDENCY_MAX_WAIT_SECS", raising=False)

    monkeypatch.setattr(rab, "read_bootstrap_heartbeat", lambda: heartbeat)

    return rab, broker


def test_runtime_dependencies_fail_fast_when_broker_absent(monkeypatch):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        rab._ensure_runtime_dependencies()

    message = str(excinfo.value)
    assert "bootstrap" in message.lower()
    assert "broker" in message.lower()
    assert "heartbeat" in message.lower()


def test_runtime_dependencies_timeout_includes_snapshot(monkeypatch):
    rab, broker = _setup_runtime_dependencies(monkeypatch, heartbeat=True, max_wait=0.02)
    broker.active_owner = True

    with pytest.raises(RuntimeError) as excinfo:
        rab._ensure_runtime_dependencies()

    message = str(excinfo.value).lower()
    assert "deadline" in message
    assert "broker snapshot" in message
    assert "heartbeat" in message


class _StubBuilder:
    def refresh_db_weights(self) -> None:
        return None


class _StubInfoDB:
    def __init__(self, path):
        self.path = path
        self.router = None
        self.apply_calls = 0

    def apply_migrations(self, **_kwargs) -> None:
        self.apply_calls += 1


def _stub_runtime(monkeypatch, rab):
    deps = types.SimpleNamespace(
        context_builder=_StubBuilder(),
        manager=types.SimpleNamespace(bootstrap_mode=False),
        data_bot=object(),
        degraded_reason=None,
        pipeline=object(),
        capability_reductions=[],
    )
    monkeypatch.setattr(rab, "_ensure_bootstrap_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(rab, "_ensure_runtime_dependencies", lambda **_k: deps)
    monkeypatch.setattr(rab, "_ensure_self_coding_decorated", lambda *_a, **_k: None)
    monkeypatch.setattr(rab, "EnhancementDB", lambda: object())


def test_deferred_migrations_latched_per_epoch(monkeypatch, tmp_path, caplog):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)
    _stub_runtime(monkeypatch, rab)
    monkeypatch.delenv(rab._INFO_DB_FORCE_REAPPLY_ENV, raising=False)
    monkeypatch.setenv(rab._INFO_DB_MIGRATION_EPOCH_ENV, "test-epoch")
    rab._DEFERRED_INFO_DB_MIGRATION_LATCH.clear()

    with caplog.at_level(logging.DEBUG, logger=rab.__name__):
        db1 = _StubInfoDB(tmp_path / "info.db")
        rab.ResearchAggregatorBot(["topic"], info_db=db1, capital_manager=object(), defer_migrations_until_ready=True)
        db2 = _StubInfoDB(tmp_path / "info.db")
        rab.ResearchAggregatorBot(["topic"], info_db=db2, capital_manager=object(), defer_migrations_until_ready=True)

    assert db1.apply_calls == 1
    assert db2.apply_calls == 0
    assert sum("migrations applied after pipeline ready" in r.message for r in caplog.records) == 1
    assert any("migrations already applied for epoch=test-epoch" in r.message for r in caplog.records)


def test_deferred_migrations_force_reapply_env(monkeypatch, tmp_path):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)
    _stub_runtime(monkeypatch, rab)
    monkeypatch.setenv(rab._INFO_DB_FORCE_REAPPLY_ENV, "1")
    monkeypatch.setenv(rab._INFO_DB_MIGRATION_EPOCH_ENV, "test-epoch-force")
    rab._DEFERRED_INFO_DB_MIGRATION_LATCH.clear()

    db1 = _StubInfoDB(tmp_path / "info.db")
    rab.ResearchAggregatorBot(["topic"], info_db=db1, capital_manager=object(), defer_migrations_until_ready=True)
    db2 = _StubInfoDB(tmp_path / "info.db")
    rab.ResearchAggregatorBot(["topic"], info_db=db2, capital_manager=object(), defer_migrations_until_ready=True)

    assert db1.apply_calls == 1
    assert db2.apply_calls == 1


def test_get_or_create_research_aggregator_reuses_instance(monkeypatch, tmp_path, caplog):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)
    _stub_runtime(monkeypatch, rab)
    monkeypatch.setenv(rab._INFO_DB_MIGRATION_EPOCH_ENV, "epoch-reuse")
    rab._CACHED_RESEARCH_AGGREGATOR = None
    rab._CACHED_RESEARCH_AGGREGATOR_EPOCH = None
    rab._CACHED_RESEARCH_AGGREGATOR_CREATED_AT = None
    rab._DEFERRED_INFO_DB_MIGRATION_LATCH.clear()

    info_db = _StubInfoDB(tmp_path / "info.db")
    with caplog.at_level(logging.DEBUG, logger=rab.__name__):
        first = rab.get_or_create_research_aggregator(
            ["topic"],
            info_db=info_db,
            context_builder=_StubBuilder(),
            capital_manager=object(),
            defer_migrations_until_ready=True,
            caller_label="test.first",
            creation_reason="unit",
        )
        second = rab.get_or_create_research_aggregator(
            ["topic"],
            info_db=info_db,
            context_builder=_StubBuilder(),
            capital_manager=object(),
            defer_migrations_until_ready=True,
            caller_label="test.second",
            creation_reason="unit",
        )

    assert first is second
    assert info_db.apply_calls == 1
    assert any(
        record.__dict__.get("instance_id") == getattr(first, "instance_id", None)
        and record.__dict__.get("caller") == "test.first"
        for record in caplog.records
    )
    assert any(
        record.__dict__.get("instance_id") == getattr(first, "instance_id", None)
        and "reused" in record.message
        for record in caplog.records
    )




def test_get_or_create_research_aggregator_backoff_after_failure(monkeypatch):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)
    _stub_runtime(monkeypatch, rab)
    monkeypatch.setenv(rab._INFO_DB_MIGRATION_EPOCH_ENV, "epoch-backoff")
    rab._CACHED_RESEARCH_AGGREGATOR = None
    rab._CACHED_RESEARCH_AGGREGATOR_EPOCH = None
    rab._CACHED_RESEARCH_AGGREGATOR_CREATED_AT = None
    rab._RESEARCH_AGGREGATOR_RETRY_ATTEMPTS = 0
    rab._RESEARCH_AGGREGATOR_RETRY_AFTER = 0.0

    class _Boom:
        def __init__(self, *_a, **_k):
            raise RuntimeError("create boom")

    monkeypatch.setattr(rab, "ResearchAggregatorBot", _Boom)

    with pytest.raises(RuntimeError, match="create boom"):
        rab.get_or_create_research_aggregator(["topic"])

    with pytest.raises(RuntimeError, match="rate-limited"):
        rab.get_or_create_research_aggregator(["topic"])
def test_get_or_create_reset_requires_explicit_event(monkeypatch):
    rab, _broker = _setup_runtime_dependencies(monkeypatch)
    monkeypatch.setenv(rab._INFO_DB_MIGRATION_EPOCH_ENV, "epoch")
    rab._CACHED_RESEARCH_AGGREGATOR = object()
    rab._CACHED_RESEARCH_AGGREGATOR_EPOCH = "epoch"
    rab._CACHED_RESEARCH_AGGREGATOR_CREATED_AT = 1.0

    rab.get_or_create_research_aggregator([], reset=True, reset_event="loop-iteration")

    assert rab._CACHED_RESEARCH_AGGREGATOR is not None
