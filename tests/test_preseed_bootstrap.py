import contextlib
import logging
import os
import sys
import time
import types

import pytest


def _install_menace_sandbox_stub() -> None:
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = []  # type: ignore[attr-defined]

    coding_stub = types.ModuleType("menace_sandbox.coding_bot_interface")
    coding_stub.logger = logging.getLogger("menace_sandbox.coding_bot_interface")
    coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT = 300.0  # type: ignore[attr-defined]

    def _get_bootstrap_wait_timeout():
        raw = os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")
        if not raw:
            return coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT

        lowered = raw.lower()
        if lowered == "none":
            return None

        try:
            parsed_timeout = float(raw)
        except ValueError:
            coding_stub.logger.warning(  # type: ignore[attr-defined]
                "Invalid MENACE_BOOTSTRAP_WAIT_SECS=%r; using default %ss",
                raw,
                coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT,
            )
            return coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT

        clamped_timeout = max(coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT, parsed_timeout)
        if clamped_timeout > parsed_timeout:
            coding_stub.logger.warning(  # type: ignore[attr-defined]
                "MENACE_BOOTSTRAP_WAIT_SECS=%r below minimum; clamping to %ss",
                raw,
                clamped_timeout,
            )
        return clamped_timeout

    def _resolve_bootstrap_wait_timeout(vector_heavy: bool = False):
        if not vector_heavy:
            raw = os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")
            if raw:
                lowered = raw.lower()
                if lowered == "none":
                    return None
                try:
                    return max(coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT, float(raw))
                except ValueError:
                    return coding_stub._MIN_BOOTSTRAP_WAIT_TIMEOUT
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
    coding_stub._get_bootstrap_wait_timeout = _get_bootstrap_wait_timeout  # type: ignore[attr-defined]
    coding_stub._BOOTSTRAP_WAIT_TIMEOUT = _get_bootstrap_wait_timeout()  # type: ignore[attr-defined]
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


def test_resolve_step_timeout_clamps_low_values(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(bootstrap, "_BASELINE_BOOTSTRAP_STEP_TIMEOUT", 60.0)
    monkeypatch.setattr(
        bootstrap._coding_bot_interface,
        "_resolve_bootstrap_wait_timeout",
        lambda vector_heavy=False: 10.0,
    )

    timeout = bootstrap._resolve_step_timeout(vector_heavy=False)

    assert timeout == pytest.approx(60.0)
    assert any("clamping" in record.message for record in caplog.records)


def test_prepare_timeout_uses_clamped_value(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    timeouts: dict[str, tuple[float | None, float | None]] = {}

    monkeypatch.setattr(bootstrap, "_BASELINE_BOOTSTRAP_STEP_TIMEOUT", 75.0)
    monkeypatch.setattr(bootstrap, "BOOTSTRAP_STEP_TIMEOUT", 75.0)
    monkeypatch.setattr(bootstrap, "_BOOTSTRAP_CACHE", {})
    monkeypatch.setattr(
        bootstrap._coding_bot_interface,
        "_resolve_bootstrap_wait_timeout",
        lambda vector_heavy=False: 10.0,
    )

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
        return object()

    monkeypatch.setattr(bootstrap, "_is_vector_bootstrap_heavy", lambda *_a, **_k: False)
    monkeypatch.setattr(bootstrap, "fallback_helper_manager", fake_fallback_helper_manager)
    monkeypatch.setattr(bootstrap, "create_context_builder", fake_context_builder)
    monkeypatch.setattr(bootstrap, "BotRegistry", lambda: object())
    monkeypatch.setattr(bootstrap, "DataBot", lambda start_server=False: object())
    monkeypatch.setattr(bootstrap, "SelfCodingEngine", lambda *_a, **_k: object())
    monkeypatch.setattr(bootstrap, "MenaceMemoryManager", lambda: object())
    monkeypatch.setattr(
        bootstrap,
        "prepare_pipeline_for_bootstrap",
        lambda **_k: (object(), lambda *_a, **_k: None),
    )
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

    def fake_run_with_timeout(fn, *, timeout: float | None, description: str, resolved_timeout=None, **kwargs):
        effective_timeout = resolved_timeout[0] if resolved_timeout else timeout
        timeouts[description] = (timeout, effective_timeout)
        if description == "prepare_pipeline_for_bootstrap":
            return (object(), lambda *_a, **_k: None)
        return None if fn is None else fn(**kwargs)

    monkeypatch.setattr(bootstrap, "_run_with_timeout", fake_run_with_timeout)

    context = bootstrap.initialize_bootstrap_context(use_cache=False)

    assert context["manager"] is dummy_manager
    assert timeouts["prepare_pipeline_for_bootstrap"] == (
        pytest.approx(bootstrap._PREPARE_SAFE_TIMEOUT_FLOOR),
        pytest.approx(bootstrap._PREPARE_SAFE_TIMEOUT_FLOOR),
    )
    assert any("clamping" in record.message for record in caplog.records)


def test_resolve_step_timeout_honors_high_defaults(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        bootstrap._coding_bot_interface,
        "_resolve_bootstrap_wait_timeout",
        lambda vector_heavy=False: 450.0,
    )

    timeout = bootstrap._resolve_step_timeout(vector_heavy=False)

    assert timeout == pytest.approx(450.0)
    assert not any("clamping" in record.message for record in caplog.records)

def test_vector_heavy_bootstrap_prefers_vector_timeout(monkeypatch, caplog):
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
    monkeypatch.setattr(bootstrap, "persist_sc_thresholds", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "ThresholdService", lambda: object())
    monkeypatch.setattr(bootstrap, "internalize_coding_bot", lambda *_a, **_k: dummy_manager)

    def fake_run_with_timeout(fn, *, timeout: float, description: str, **kwargs):
        timeouts[description] = timeout
        if description == "prepare_pipeline_for_bootstrap":
            return (object(), lambda *_a, **_k: None)
        return None if fn is None else fn(**kwargs)

    monkeypatch.setattr(bootstrap, "_run_with_timeout", fake_run_with_timeout)

    caplog.set_level(logging.WARNING)

    context = bootstrap.initialize_bootstrap_context(use_cache=False)

    assert context["manager"] is dummy_manager
    expected_timeout = max(120.0, bootstrap._BASELINE_BOOTSTRAP_STEP_TIMEOUT)
    assert timeouts["prepare_pipeline_for_bootstrap"] == pytest.approx(expected_timeout)
    if expected_timeout > 120.0:
        assert any("clamping" in record.message for record in caplog.records)
    assert timeouts["_seed_research_aggregator_context placeholder"] == pytest.approx(30.0)


def test_menace_bootstrap_wait_is_clamped(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "120")

    clamped_timeout = bootstrap._coding_bot_interface._get_bootstrap_wait_timeout()
    bootstrap._coding_bot_interface._BOOTSTRAP_WAIT_TIMEOUT = clamped_timeout

    bootstrap._BASELINE_BOOTSTRAP_STEP_TIMEOUT = max(
        300.0,
        (
            bootstrap._coding_bot_interface._BOOTSTRAP_WAIT_TIMEOUT
            if getattr(bootstrap._coding_bot_interface, "_BOOTSTRAP_WAIT_TIMEOUT", None)
            is not None
            else 300.0
        ),
    )
    bootstrap._DEFAULT_BOOTSTRAP_STEP_TIMEOUT = max(
        bootstrap._BASELINE_BOOTSTRAP_STEP_TIMEOUT,
        float(
            os.getenv(
                "BOOTSTRAP_STEP_TIMEOUT", str(bootstrap._BASELINE_BOOTSTRAP_STEP_TIMEOUT)
            )
        ),
    )
    bootstrap.BOOTSTRAP_STEP_TIMEOUT = bootstrap._resolve_step_timeout()

    assert clamped_timeout == pytest.approx(300.0)
    assert bootstrap._BASELINE_BOOTSTRAP_STEP_TIMEOUT == pytest.approx(clamped_timeout)
    assert bootstrap.BOOTSTRAP_STEP_TIMEOUT == pytest.approx(clamped_timeout)
    assert any("below minimum" in record.message for record in caplog.records)


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
