import threading
import time

import pytest

from unit_tests.test_preseed_bootstrap_timeout_safety import (
    _install_stub_module,
    _load_preseed_bootstrap_module,
)


def test_embedder_thread_cancelled_when_budget_missing():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def get_embedder(*_, stop_event: threading.Event, **__):
        calls["worker_started"] = time.perf_counter()
        stop_event.wait(5)
        calls["worker_unblocked"] = time.perf_counter()
        calls["stop_event_state"] = stop_event.is_set()
        return None

    def cancel_embedder_initialisation(stop_event: threading.Event, **__):
        calls["cancel_called"] = time.perf_counter()
        calls["cancel_stop_event"] = stop_event
        stop_event.set()

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": 0.2,
            "_activate_bundled_fallback": lambda *_args, **_kwargs: False,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.2,
            "cancel_embedder_initialisation": cancel_embedder_initialisation,
            "embedder_cache_present": lambda *_args, **_kwargs: False,
            "get_embedder": get_embedder,
        },
    )

    # Reset shared flags for a deterministic run.
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    start = time.perf_counter()
    preseed_bootstrap._bootstrap_embedder(timeout=0.05)
    elapsed = time.perf_counter() - start

    for _ in range(10):
        if calls.get("worker_unblocked"):
            break
        time.sleep(0.05)

    cancel_event = calls.get("cancel_stop_event")

    assert elapsed < 0.5
    assert isinstance(cancel_event, threading.Event)
    assert cancel_event.is_set()
    assert calls.get("cancel_called") is not None
    assert calls.get("worker_unblocked") is not None
    assert calls.get("worker_unblocked") - start < 0.5
    assert calls.get("stop_event_state") is True
    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("thread") is None
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER


def test_embedder_cancellation_signals_budget_and_fallback():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    class Budget:
        def record_progress(self, label: str, *, elapsed: float, remaining: float | None, metadata=None):
            calls["budget_progress"] = (label, elapsed, remaining, dict(metadata or {}))

        def mark_component_state(self, label: str, state: str):
            calls["component_state"] = (label, state)

    def get_embedder(*_, stop_event: threading.Event, **__):
        calls["worker_started"] = time.perf_counter()
        stop_event.wait(5)
        calls["worker_unblocked"] = time.perf_counter()
        return None

    def cancel_embedder_initialisation(stop_event: threading.Event, **__):
        calls["cancel_called"] = time.perf_counter()
        stop_event.set()

    def activate_bundled_fallback(reason: str):
        calls["fallback_reason"] = reason
        calls["fallback_activated"] = True
        return preseed_bootstrap._BOOTSTRAP_PLACEHOLDER

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": -1.0,
            "_activate_bundled_fallback": activate_bundled_fallback,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.05,
            "cancel_embedder_initialisation": cancel_embedder_initialisation,
            "embedder_cache_present": lambda *_args, **_kwargs: False,
            "get_embedder": get_embedder,
        },
    )

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    budget = Budget()
    start = time.perf_counter()
    preseed_bootstrap._bootstrap_embedder(
        timeout=0.2,
        stage_budget=None,
        budget=budget,
        budget_label="vector_seeding",
    )
    elapsed = time.perf_counter() - start

    for _ in range(20):
        if calls.get("cancel_called"):
            break
        time.sleep(0.05)

    assert calls.get("cancel_called") is not None
    assert calls.get("worker_unblocked") is not None
    assert calls.get("fallback_activated") is True
    assert calls.get("fallback_reason") == "bootstrap_wall_clock_exceeded"
    assert calls.get("component_state") == ("vector_seeding", "blocked")
    assert calls.get("budget_progress", (None, None, None, {}))[0] == "vector_seeding"
    assert calls.get("budget_progress", (None, None, None, {}))[2] == 0.0
    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("thread") is None
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert elapsed < 0.5


def test_embedder_presence_probe_records_placeholder_without_background_download():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def get_embedder(*_, **__):
        calls["background_get"] = time.perf_counter()
        return object()

    def embedder_cache_present():
        calls["cache_probe"] = time.perf_counter()
        return True

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": 0.2,
            "_activate_bundled_fallback": lambda *_args, **_kwargs: False,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.1,
            "cancel_embedder_initialisation": lambda *_args, **_kwargs: None,
            "embedder_cache_present": embedder_cache_present,
            "get_embedder": get_embedder,
        },
    )

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    preseed_bootstrap._bootstrap_embedder(
        timeout=0.05,
        stage_budget=0.1,
        presence_probe=True,
        presence_reason="embedder_presence_probe",
        bootstrap_fast=True,
    )

    for _ in range(20):
        if calls.get("background_get"):
            break
        time.sleep(0.05)

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert calls.get("cache_probe") is not None
    assert job.get("deferred") is True
    assert job.get("placeholder_reason") == "embedder_presence_probe"
    assert job.get("presence_available") is True
    assert job.get("background_scheduled") is None
    assert calls.get("background_get") is None
    assert job.get("thread") is None


def test_presence_probe_returns_placeholder_immediately():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    def embedder_cache_present():
        return True

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": 0.2,
            "_activate_bundled_fallback": lambda *_args, **_kwargs: False,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.1,
            "cancel_embedder_initialisation": lambda *_args, **_kwargs: None,
            "embedder_cache_present": embedder_cache_present,
            "get_embedder": lambda *_args, **_kwargs: object(),
        },
    )

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None
    placeholder = preseed_bootstrap._bootstrap_embedder(
        timeout=0.05,
        stage_budget=0.1,
        presence_probe=True,
        presence_reason="embedder_presence_guard",
        bootstrap_fast=True,
    )

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert placeholder == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("placeholder_reason") == "embedder_presence_guard"
    assert job.get("thread") is None
    assert job.get("deferred") is True
    assert job.get("presence_available") is True


def test_embedder_stage_budget_sets_deferral_token():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def get_embedder(*_, stop_event: threading.Event, **__):
        calls["worker_started"] = time.perf_counter()
        stop_event.wait(1)
        return None

    def cancel_embedder_initialisation(stop_event: threading.Event, **__):
        calls["cancel_called"] = time.perf_counter()
        stop_event.set()

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": 0.5,
            "_activate_bundled_fallback": lambda *_args, **_kwargs: False,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.5,
            "cancel_embedder_initialisation": cancel_embedder_initialisation,
            "embedder_cache_present": lambda *_args, **_kwargs: False,
            "get_embedder": get_embedder,
        },
    )

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False

    preseed_bootstrap._bootstrap_embedder(
        timeout=0.5,
        stage_budget=0.05,
        budget_label="vector_seeding",
    )

    for _ in range(20):
        job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
        if job.get("deferred"):
            break
        time.sleep(0.05)

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("deferred") is True
    assert job.get("placeholder_reason") in {
        "bootstrap_budget_exceeded",
        "bootstrap_wall_clock_exceeded",
        "stage_wall_cap_exceeded",
        "embedder_max_duration_guard",
    }
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    for _ in range(10):
        if calls.get("cancel_called"):
            break
        time.sleep(0.05)

    assert calls.get("cancel_called") is not None


def test_existing_download_cancelled_with_stage_caps():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    stop_event = threading.Event()
    join_calls: dict[str, object] = {}

    def cancel_embedder_initialisation(stop_event: threading.Event, **kwargs):
        join_calls["cancel_args"] = kwargs
        stop_event.set()

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": 0.5,
            "_activate_bundled_fallback": lambda *_args, **_kwargs: False,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.25,
            "cancel_embedder_initialisation": cancel_embedder_initialisation,
            "embedder_cache_present": lambda *_args, **_kwargs: False,
            "get_embedder": lambda *_args, **_kwargs: object(),
        },
    )

    def _background_worker():
        stop_event.wait(5)

    existing_thread = threading.Thread(target=_background_worker)
    existing_thread.start()

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = {
        "thread": existing_thread,
        "stop_event": stop_event,
        "placeholder": preseed_bootstrap._BOOTSTRAP_PLACEHOLDER,
    }

    placeholder = preseed_bootstrap._bootstrap_embedder(timeout=0.2, stage_budget=0.1)

    existing_thread.join(0.5)
    assert not existing_thread.is_alive()
    assert stop_event.is_set()
    assert join_calls["cancel_args"].get("join_timeout") == 0.1
    assert placeholder == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB is None


def test_warmup_uses_derived_join_cap(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def slow_bootstrap_embedder(*_, stop_event: threading.Event, **__):
        calls["start"] = time.perf_counter()
        stop_event.wait(5)
        calls["end"] = time.perf_counter()
        calls["stop_state"] = stop_event.is_set()
        return object()

    monkeypatch.setattr(
        preseed_bootstrap, "_bootstrap_embedder", slow_bootstrap_embedder
    )
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_budget", lambda *_, **__: 5.0
    )
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_deadline", lambda *_, **__: None
    )
    monkeypatch.setattr(
        preseed_bootstrap,
        "_derive_warmup_join_timeout",
        lambda **__: (0.1, 0.1),
    )

    class DummyCoordinator:
        def negotiate_step(self, *_args, **_kwargs):
            return {"timeout_scale": 1.0, "contention": False}

    monkeypatch.setattr(
        preseed_bootstrap, "_BOOTSTRAP_CONTENTION_COORDINATOR", DummyCoordinator()
    )

    preseed_bootstrap._BOOTSTRAP_CACHE.clear()
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    start = time.perf_counter()
    preseed_bootstrap.initialize_bootstrap_context(
        use_cache=False, full_embedder_preload=True
    )
    elapsed = time.perf_counter() - start

    for _ in range(40):
        if calls.get("end"):
            break
        time.sleep(0.05)

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}

    assert elapsed < 2.0
    assert calls.get("start") is not None
    assert calls.get("end") is not None
    assert calls.get("stop_state") is True
    assert calls["end"] - calls["start"] < 0.5
    assert job.get("deferred") is True
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("deferral_reason") == "embedder_preload_warmup_cap_exceeded"
    assert job.get("strict_timebox") == 0.1


def test_warmup_join_timeout_capped(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    monkeypatch.setattr(preseed_bootstrap, "BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP", 0.5)

    start = time.monotonic()
    join_timeout, join_cap = preseed_bootstrap._derive_warmup_join_timeout(
        warmup_timebox_cap=5.0,
        enforced_timebox=10.0,
        warmup_started=start,
        stage_guard_timebox=None,
        embedder_stage_budget_hint=None,
        warmup_hard_cap=None,
    )

    assert join_timeout == 0.5
    assert join_cap == 0.5


def test_start_embedder_warmup_aborts_on_elapsed_deadline(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def _bootstrap_embedder(*_args, **_kwargs):
        calls["bootstrap_invoked"] = True
        return object()

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", _bootstrap_embedder)
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    result = preseed_bootstrap.start_embedder_warmup(
        stage_budget=0.1, bootstrap_deadline=time.perf_counter() - 0.01
    )

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert result == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert "bootstrap_invoked" not in calls
    assert job.get("deferral_reason") == "embedder_bootstrap_deadline_elapsed"
    assert job.get("full_preload_skipped") is True
    summary = job.get("warmup_summary") or {}
    assert summary.get("full_preload_skipped") is True


def test_start_embedder_warmup_skips_on_tight_budget_window(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    calls: dict[str, object] = {}

    def _bootstrap_embedder(*_args, **_kwargs):
        calls["bootstrap_invoked"] = True
        return object()

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", _bootstrap_embedder)
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    result = preseed_bootstrap.start_embedder_warmup(
        stage_budget=0.005, bootstrap_deadline=time.perf_counter() + 0.005
    )

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}

    assert result == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert "bootstrap_invoked" not in calls
    assert job.get("deferral_reason") == "embedder_budget_window_too_short"
    assert job.get("background_join_timeout") is not None


def test_start_embedder_warmup_caps_background_join(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    join_seen = threading.Event()

    def background_task():
        join_seen.set()
        time.sleep(0.2)

    future = preseed_bootstrap._BOOTSTRAP_BACKGROUND_EXECUTOR.submit(background_task)
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = {
        "placeholder": preseed_bootstrap._BOOTSTRAP_PLACEHOLDER,
        "deferred": True,
        "ready_after_bootstrap": True,
        "placeholder_reason": "embedder_budget_guarded",
        "background_join_timeout": 10.0,
        "background_future": future,
    }

    start = time.perf_counter()
    result = preseed_bootstrap.start_embedder_warmup(
        timeout=1.0,
        stage_budget=0.05,
        bootstrap_deadline=time.perf_counter() + 0.05,
    )
    elapsed = time.perf_counter() - start

    assert result is preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert elapsed < 0.2
    assert join_seen.wait(1.0)


def test_start_embedder_warmup_records_full_preload_deferral(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", pytest.fail)
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None

    result = preseed_bootstrap.start_embedder_warmup(
        stage_budget=None, bootstrap_deadline=time.perf_counter() + 5.0
    )

    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert result == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("deferral_reason") == "embedder_stage_budget_missing"
    assert job.get("full_preload_skipped") is True
    summary = job.get("warmup_summary") or {}
    assert summary.get("deferred") is True
    assert summary.get("full_preload_skipped") is True

