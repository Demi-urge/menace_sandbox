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
            "get_embedder": get_embedder,
        },
    )

    # Reset shared flags for a deterministic run.
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False

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
        return True

    _install_stub_module(
        "menace_sandbox.governed_embeddings",
        {
            "_MAX_EMBEDDER_WAIT": -1.0,
            "_activate_bundled_fallback": activate_bundled_fallback,
            "apply_bootstrap_timeout_caps": lambda *_args, **_kwargs: 0.05,
            "cancel_embedder_initialisation": cancel_embedder_initialisation,
            "get_embedder": get_embedder,
        },
    )

    preseed_bootstrap._BOOTSTRAP_EMBEDDER_DISABLED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_ATTEMPTED = False
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_STARTED = False

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
    assert elapsed < 0.5

