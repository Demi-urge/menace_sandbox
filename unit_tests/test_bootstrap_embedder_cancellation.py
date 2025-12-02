import time
import threading

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
    }
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    for _ in range(10):
        if calls.get("cancel_called"):
            break
        time.sleep(0.05)

    assert calls.get("cancel_called") is not None

