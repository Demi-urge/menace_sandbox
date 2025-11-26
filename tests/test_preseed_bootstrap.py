import contextlib
import os
import sys
import time
import types

import pytest


def _install_menace_sandbox_stub() -> None:
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = []  # type: ignore[attr-defined]

    coding_stub = types.ModuleType("menace_sandbox.coding_bot_interface")
    def _resolve_bootstrap_wait_timeout(vector_heavy: bool = False):
        if not vector_heavy:
            raw = os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")
            if raw:
                lowered = raw.lower()
                if lowered == "none":
                    return None
                try:
                    return max(1.0, float(raw))
                except ValueError:
                    return 300.0
            return coding_stub._BOOTSTRAP_WAIT_TIMEOUT

        raw = os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS")
        default_timeout = 900.0
        if raw:
            lowered = raw.lower()
            if lowered == "none":
                return None
            try:
                return max(1.0, float(raw))
            except ValueError:
                return default_timeout
        if coding_stub._BOOTSTRAP_WAIT_TIMEOUT is None:
            return default_timeout
        return max(coding_stub._BOOTSTRAP_WAIT_TIMEOUT, default_timeout)

    coding_stub._resolve_bootstrap_wait_timeout = _resolve_bootstrap_wait_timeout  # type: ignore[attr-defined]
    coding_stub._BOOTSTRAP_WAIT_TIMEOUT = 300.0  # type: ignore[attr-defined]
    coding_stub._push_bootstrap_context = lambda **_k: {"placeholder": True}  # type: ignore[attr-defined]
    coding_stub._pop_bootstrap_context = lambda *_a, **_k: None  # type: ignore[attr-defined]

    @contextlib.contextmanager
    def _helper_manager(**_kwargs):
        yield types.SimpleNamespace(bootstrap_runtime_active=True)

    coding_stub.fallback_helper_manager = _helper_manager  # type: ignore[attr-defined]
    coding_stub.prepare_pipeline_for_bootstrap = (  # type: ignore[attr-defined]
        lambda **_k: (object(), lambda *_a, **_k: None)
    )

    def _simple_class(_name: str):
        return type(_name, (), {})

    bot_registry = types.ModuleType("menace_sandbox.bot_registry")
    bot_registry.BotRegistry = _simple_class("BotRegistry")

    code_db = types.ModuleType("menace_sandbox.code_database")
    code_db.CodeDB = _simple_class("CodeDB")

    context_builder_util = types.ModuleType("menace_sandbox.context_builder_util")
    context_builder_util.create_context_builder = lambda **_k: object()

    db_router = types.ModuleType("menace_sandbox.db_router")
    db_router.set_audit_bootstrap_safe_default = lambda *_a, **_k: None

    data_bot = types.ModuleType("menace_sandbox.data_bot")
    data_bot.DataBot = _simple_class("DataBot")
    data_bot.persist_sc_thresholds = lambda *_a, **_k: None

    menace_memory_manager = types.ModuleType("menace_sandbox.menace_memory_manager")
    menace_memory_manager.MenaceMemoryManager = _simple_class("MenaceMemoryManager")

    model_pipeline = types.ModuleType("menace_sandbox.model_automation_pipeline")
    model_pipeline.ModelAutomationPipeline = _simple_class("ModelAutomationPipeline")

    self_coding_engine = types.ModuleType("menace_sandbox.self_coding_engine")
    self_coding_engine.SelfCodingEngine = _simple_class("SelfCodingEngine")

    self_coding_manager = types.ModuleType("menace_sandbox.self_coding_manager")
    self_coding_manager.SelfCodingManager = _simple_class("SelfCodingManager")
    self_coding_manager.internalize_coding_bot = lambda *_a, **_k: None

    thresholds = types.ModuleType("menace_sandbox.self_coding_thresholds")
    thresholds.get_thresholds = lambda *_a, **_k: None

    threshold_service = types.ModuleType("menace_sandbox.threshold_service")
    threshold_service.ThresholdService = _simple_class("ThresholdService")

    modules = {
        "menace_sandbox": pkg,
        "menace_sandbox.coding_bot_interface": coding_stub,
        "menace_sandbox.bot_registry": bot_registry,
        "menace_sandbox.code_database": code_db,
        "menace_sandbox.context_builder_util": context_builder_util,
        "menace_sandbox.db_router": db_router,
        "menace_sandbox.data_bot": data_bot,
        "menace_sandbox.menace_memory_manager": menace_memory_manager,
        "menace_sandbox.model_automation_pipeline": model_pipeline,
        "menace_sandbox.self_coding_engine": self_coding_engine,
        "menace_sandbox.self_coding_manager": self_coding_manager,
        "menace_sandbox.self_coding_thresholds": thresholds,
        "menace_sandbox.threshold_service": threshold_service,
    }

    for name, module in modules.items():
        sys.modules.setdefault(name, module)
    pkg.coding_bot_interface = coding_stub
    pkg.bot_registry = bot_registry
    pkg.code_database = code_db
    pkg.context_builder_util = context_builder_util
    pkg.db_router = db_router
    pkg.data_bot = data_bot
    pkg.menace_memory_manager = menace_memory_manager
    pkg.model_automation_pipeline = model_pipeline
    pkg.self_coding_engine = self_coding_engine
    pkg.self_coding_manager = self_coding_manager
    pkg.self_coding_thresholds = thresholds
    pkg.threshold_service = threshold_service


_install_menace_sandbox_stub()

import sandbox.preseed_bootstrap as bootstrap
from sandbox.preseed_bootstrap import _run_with_timeout


@pytest.fixture(autouse=True)
def reset_bootstrap_timeline():
    bootstrap.BOOTSTRAP_STEP_TIMELINE.clear()
    bootstrap._BOOTSTRAP_TIMELINE_START = None
    bootstrap.BOOTSTRAP_PROGRESS["last_step"] = "not-started"
    yield
    bootstrap.BOOTSTRAP_STEP_TIMELINE.clear()
    bootstrap._BOOTSTRAP_TIMELINE_START = None
    bootstrap.BOOTSTRAP_PROGRESS["last_step"] = "not-started"


def test_run_with_timeout_respects_requested_timeout():
    """Ensure bootstrap helpers don't stretch timeouts to the overall deadline."""

    requested_timeout = 0.1
    deadline = time.monotonic() + 1.0

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        _run_with_timeout(
            lambda: time.sleep(0.5),
            timeout=requested_timeout,
            bootstrap_deadline=deadline,
            description="sleepy",
        )

    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_run_with_timeout_emits_metadata(capsys):
    def sleepy_task(duration: float) -> None:
        time.sleep(duration)

    with pytest.raises(TimeoutError):
        _run_with_timeout(
            sleepy_task,
            timeout=0.05,
            bootstrap_deadline=time.monotonic() + 5.0,
            description="metadata-check",
            duration=0.2,
        )

    captured = capsys.readouterr().out
    assert "[bootstrap-timeout][metadata]" in captured
    assert "metadata-check" in captured
    assert "[bootstrap-timeout][thread=" in captured


def test_run_with_timeout_reports_timeline(capsys):
    bootstrap._mark_bootstrap_step("phase-one")
    time.sleep(0.01)
    bootstrap._mark_bootstrap_step("phase-two")

    with pytest.raises(TimeoutError):
        _run_with_timeout(
            lambda: time.sleep(0.2),
            timeout=0.05,
            description="timeline-check",
        )

    captured = capsys.readouterr().out
    assert "active_step=phase-two" in captured
    assert "[bootstrap-timeout][timeline] phase-one" in captured
    assert "[bootstrap-timeout][timeline] phase-two" in captured


def test_vector_heavy_bootstrap_prefers_vector_timeout(monkeypatch):
    """Vector-heavy bootstraps should inherit the extended wait window."""

    timeouts: dict[str, float] = {}

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "30")
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "120")
    monkeypatch.setattr(bootstrap._coding_bot_interface, "_BOOTSTRAP_WAIT_TIMEOUT", 30.0)
    monkeypatch.setattr(bootstrap, "BOOTSTRAP_STEP_TIMEOUT", 30.0)
    monkeypatch.setattr(bootstrap, "_BOOTSTRAP_CACHE", {})

    class DummyManager:
        engine = "engine"
        bootstrap_runtime_active = True

        def __call__(self, *_args, **_kwargs):  # pragma: no cover - compatibility hook
            return self

    dummy_manager = DummyManager()

    @contextlib.contextmanager
    def fake_fallback_helper_manager(**_kwargs):
        yield dummy_manager

    def fake_context_builder(**_kwargs):
        class _VectorHeavyBuilder:
            __module__ = "vector_service.bootstrap"

        return _VectorHeavyBuilder()

    monkeypatch.setattr(bootstrap, "fallback_helper_manager", fake_fallback_helper_manager)
    monkeypatch.setattr(bootstrap, "create_context_builder", fake_context_builder)
    monkeypatch.setattr(bootstrap, "BotRegistry", lambda: object())
    monkeypatch.setattr(bootstrap, "DataBot", lambda start_server=False: object())
    monkeypatch.setattr(bootstrap, "SelfCodingEngine", lambda *_a, **_k: object())
    monkeypatch.setattr(bootstrap, "MenaceMemoryManager", lambda: object())
    monkeypatch.setattr(bootstrap, "prepare_pipeline_for_bootstrap", lambda **_k: (object(), lambda *_a, **_k: None))
    monkeypatch.setattr(bootstrap, "_push_bootstrap_context", lambda **_k: {"placeholder": True})
    monkeypatch.setattr(bootstrap, "_pop_bootstrap_context", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "_seed_research_aggregator_context", lambda **_k: None)
    monkeypatch.setattr(bootstrap, "get_thresholds", lambda *_a, **_k: type("Thresholds", (), {
        "roi_drop": 1,
        "error_increase": 2,
        "test_failure_increase": 3,
    })())
    monkeypatch.setattr(bootstrap, "persist_sc_thresholds", lambda **_k: None)
    monkeypatch.setattr(bootstrap, "ThresholdService", lambda: object())
    monkeypatch.setattr(bootstrap, "internalize_coding_bot", lambda *_a, **_k: dummy_manager)

    def fake_run_with_timeout(fn, *, timeout: float, description: str, **kwargs):
        timeouts[description] = timeout
        if description == "prepare_pipeline_for_bootstrap":
            return (object(), lambda *_a, **_k: None)
        return None if fn is None else fn(**kwargs)

    monkeypatch.setattr(bootstrap, "_run_with_timeout", fake_run_with_timeout)

    context = bootstrap.initialize_bootstrap_context(use_cache=False)

    assert context["manager"] is dummy_manager
    assert timeouts["prepare_pipeline_for_bootstrap"] == pytest.approx(120.0)
    assert timeouts["_seed_research_aggregator_context placeholder"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    "resolver_behavior",
    (
        lambda *_a, **_k: None,
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("resolver boom")),
    ),
)
def test_prepare_pipeline_timeout_falls_back_when_resolver_none(monkeypatch, resolver_behavior):
    """Prepare should honor fallback timeouts when the resolver cannot supply one."""

    timeouts: dict[str, float | None] = {}

    monkeypatch.setattr(bootstrap._coding_bot_interface, "_resolve_bootstrap_wait_timeout", resolver_behavior)
    monkeypatch.setattr(bootstrap._coding_bot_interface, "_BOOTSTRAP_WAIT_TIMEOUT", None)
    monkeypatch.setattr(bootstrap, "_BOOTSTRAP_CACHE", {})

    class DummyManager:
        engine = "engine"
        bootstrap_runtime_active = True

        def __call__(self, *_args, **_kwargs):  # pragma: no cover - compatibility hook
            return self

    dummy_manager = DummyManager()

    @contextlib.contextmanager
    def fake_fallback_helper_manager(**_kwargs):
        yield dummy_manager

    def fake_context_builder(**_kwargs):
        class _VectorHeavyBuilder:
            __module__ = "vector_service.bootstrap"

        return _VectorHeavyBuilder()

    monkeypatch.setattr(bootstrap, "fallback_helper_manager", fake_fallback_helper_manager)
    monkeypatch.setattr(bootstrap, "create_context_builder", fake_context_builder)
    monkeypatch.setattr(bootstrap, "BotRegistry", lambda: object())
    monkeypatch.setattr(bootstrap, "DataBot", lambda start_server=False: object())
    monkeypatch.setattr(bootstrap, "SelfCodingEngine", lambda *_a, **_k: object())
    monkeypatch.setattr(bootstrap, "MenaceMemoryManager", lambda: object())
    monkeypatch.setattr(bootstrap, "prepare_pipeline_for_bootstrap", lambda **_k: (object(), lambda *_a, **_k: None))
    monkeypatch.setattr(bootstrap, "_push_bootstrap_context", lambda **_k: {"placeholder": True})
    monkeypatch.setattr(bootstrap, "_pop_bootstrap_context", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "_seed_research_aggregator_context", lambda **_k: None)
    monkeypatch.setattr(bootstrap, "get_thresholds", lambda *_a, **_k: type("Thresholds", (), {
        "roi_drop": 1,
        "error_increase": 2,
        "test_failure_increase": 3,
    })())
    monkeypatch.setattr(bootstrap, "persist_sc_thresholds", lambda **_k: None)
    monkeypatch.setattr(bootstrap, "ThresholdService", lambda: object())
    monkeypatch.setattr(bootstrap, "internalize_coding_bot", lambda *_a, **_k: dummy_manager)
    monkeypatch.setattr(bootstrap, "_is_vector_bootstrap_heavy", lambda *_a, **_k: True)

    def fake_run_with_timeout(fn, *, timeout: float | None, description: str, **kwargs):
        timeouts[description] = timeout
        if description == "prepare_pipeline_for_bootstrap":
            return (object(), lambda *_a, **_k: None)
        return None if fn is None else fn(**kwargs)

    monkeypatch.setattr(bootstrap, "_run_with_timeout", fake_run_with_timeout)

    bootstrap.BOOTSTRAP_STEP_TIMEOUT = bootstrap._resolve_step_timeout()

    context = bootstrap.initialize_bootstrap_context(use_cache=False)

    assert context["manager"] is dummy_manager
    assert timeouts["prepare_pipeline_for_bootstrap"] == pytest.approx(bootstrap._DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT)
    assert timeouts["_seed_research_aggregator_context placeholder"] == pytest.approx(
        bootstrap._DEFAULT_BOOTSTRAP_STEP_TIMEOUT
    )
