import contextlib
import time

import pytest

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
    deadline = time.monotonic() + 10.0

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


def test_prepare_pipeline_timeout_respects_none(monkeypatch):
    """Disabling bootstrap timeouts should flow through to prepare pipeline."""

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "none")
    monkeypatch.setattr(bootstrap._coding_bot_interface, "_BOOTSTRAP_WAIT_TIMEOUT", None)
    monkeypatch.setattr(
        bootstrap._coding_bot_interface,
        "_resolve_bootstrap_wait_timeout",
        lambda vector_heavy=False: None,
    )

    bootstrap.BOOTSTRAP_STEP_TIMEOUT = bootstrap._resolve_step_timeout()
    effective_timeout, timeout_context = bootstrap._resolve_timeout(
        bootstrap.BOOTSTRAP_STEP_TIMEOUT,
        bootstrap_deadline=None,
        heavy_bootstrap=False,
    )

    assert bootstrap.BOOTSTRAP_STEP_TIMEOUT is None
    assert effective_timeout is None
    assert timeout_context["effective_timeout"] is None

    def slow_prepare(**_kwargs):
        time.sleep(0.05)
        return "pipeline", lambda *_a, **_k: None

    pipeline, _promote = bootstrap._run_with_timeout(
        slow_prepare,
        timeout=bootstrap.BOOTSTRAP_STEP_TIMEOUT,
        description="prepare_pipeline_for_bootstrap",
        resolved_timeout=(effective_timeout, timeout_context),
    )

    assert pipeline == "pipeline"
