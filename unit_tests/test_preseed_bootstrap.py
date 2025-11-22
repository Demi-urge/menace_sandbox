import logging
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import sandbox.preseed_bootstrap as preseed


def test_prepare_pipeline_timeout_respects_configured_cap(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    placeholder_context = object()
    recorded_timeouts: dict[str, float] = {}

    class DummyManager:
        pass

    class DummyPipeline:
        pass

    def fake_run_with_timeout(fn, *, timeout, description, **kwargs):
        recorded_timeouts[description] = timeout
        return fn(**kwargs)

    @contextmanager
    def dummy_fallback_helper_manager(**kwargs):
        yield DummyManager()

    def fake_prepare_pipeline_for_bootstrap(**kwargs):
        return DummyPipeline(), (lambda **promo_kwargs: None)

    monkeypatch.setattr(preseed, "_bootstrap_embedder", lambda *a, **kw: None)
    monkeypatch.setattr(preseed, "_run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(preseed, "fallback_helper_manager", dummy_fallback_helper_manager)
    monkeypatch.setattr(preseed, "create_context_builder", lambda: "builder")
    monkeypatch.setattr(preseed, "BotRegistry", lambda: "registry")
    monkeypatch.setattr(preseed, "DataBot", lambda start_server=False: "data_bot")
    monkeypatch.setattr(preseed, "SelfCodingEngine", lambda *a, **kw: "engine")
    monkeypatch.setattr(preseed, "ModelAutomationPipeline", DummyPipeline)
    monkeypatch.setattr(preseed, "MenaceMemoryManager", lambda: "memory")
    monkeypatch.setattr(preseed, "CodeDB", lambda: "code_db")
    monkeypatch.setattr(preseed, "ThresholdService", lambda: "threshold_service")
    monkeypatch.setattr(preseed, "prepare_pipeline_for_bootstrap", fake_prepare_pipeline_for_bootstrap)
    monkeypatch.setattr(preseed, "internalize_coding_bot", lambda *a, **kw: DummyManager())
    monkeypatch.setattr(preseed, "_push_bootstrap_context", lambda **kwargs: placeholder_context)
    monkeypatch.setattr(preseed, "_pop_bootstrap_context", lambda ctx: None)
    monkeypatch.setattr(preseed, "_seed_research_aggregator_context", lambda **kwargs: None)
    monkeypatch.setattr(preseed, "persist_sc_thresholds", lambda *a, **kw: None)
    monkeypatch.setattr(
        preseed,
        "get_thresholds",
        lambda bot_name: SimpleNamespace(
            roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0
        ),
    )

    bootstrap_deadline = time.monotonic() + 500.0
    configured_timeout = 5.0
    preseed.initialize_bootstrap_context(
        bot_name="TestBot",
        use_cache=False,
        bootstrap_deadline=bootstrap_deadline,
        prepare_pipeline_timeout=configured_timeout,
    )

    selected_timeout = recorded_timeouts["prepare_pipeline_for_bootstrap"]
    assert selected_timeout == configured_timeout
    assert "prepare_pipeline_for_bootstrap timeout selected" in caplog.text

    log_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "prepare_pipeline_for_bootstrap timeout selected"
    )
    assert getattr(log_record, "timeout_source", None) == "configured_override"


def test_prepare_pipeline_timeout_defaults_to_bootstrap_deadline(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    placeholder_context = object()
    recorded_timeouts: dict[str, float] = {}

    class DummyManager:
        pass

    class DummyPipeline:
        pass

    def fake_run_with_timeout(fn, *, timeout, description, **kwargs):
        recorded_timeouts[description] = timeout
        return fn(**kwargs)

    @contextmanager
    def dummy_fallback_helper_manager(**kwargs):
        yield DummyManager()

    def fake_prepare_pipeline_for_bootstrap(**kwargs):
        return DummyPipeline(), (lambda **promo_kwargs: None)

    monkeypatch.setattr(preseed, "_bootstrap_embedder", lambda *a, **kw: None)
    monkeypatch.setattr(preseed, "_run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(preseed, "fallback_helper_manager", dummy_fallback_helper_manager)
    monkeypatch.setattr(preseed, "create_context_builder", lambda: "builder")
    monkeypatch.setattr(preseed, "BotRegistry", lambda: "registry")
    monkeypatch.setattr(preseed, "DataBot", lambda start_server=False: "data_bot")
    monkeypatch.setattr(preseed, "SelfCodingEngine", lambda *a, **kw: "engine")
    monkeypatch.setattr(preseed, "ModelAutomationPipeline", DummyPipeline)
    monkeypatch.setattr(preseed, "MenaceMemoryManager", lambda: "memory")
    monkeypatch.setattr(preseed, "CodeDB", lambda: "code_db")
    monkeypatch.setattr(preseed, "ThresholdService", lambda: "threshold_service")
    monkeypatch.setattr(preseed, "prepare_pipeline_for_bootstrap", fake_prepare_pipeline_for_bootstrap)
    monkeypatch.setattr(preseed, "internalize_coding_bot", lambda *a, **kw: DummyManager())
    monkeypatch.setattr(preseed, "_push_bootstrap_context", lambda **kwargs: placeholder_context)
    monkeypatch.setattr(preseed, "_pop_bootstrap_context", lambda ctx: None)
    monkeypatch.setattr(preseed, "_seed_research_aggregator_context", lambda **kwargs: None)
    monkeypatch.setattr(preseed, "persist_sc_thresholds", lambda *a, **kw: None)
    monkeypatch.setattr(
        preseed,
        "get_thresholds",
        lambda bot_name: SimpleNamespace(
            roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0
        ),
    )

    base_time = 1000.0
    monkeypatch.setattr(preseed.time, "monotonic", lambda: base_time)
    bootstrap_deadline = base_time + 42.0

    preseed.initialize_bootstrap_context(
        bot_name="TestBot",
        use_cache=False,
        bootstrap_deadline=bootstrap_deadline,
        prepare_pipeline_timeout=None,
    )

    selected_timeout = recorded_timeouts["prepare_pipeline_for_bootstrap"]
    assert selected_timeout == pytest.approx(42.0)
    log_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "prepare_pipeline_for_bootstrap timeout selected"
    )
    assert getattr(log_record, "timeout_source", None) == "bootstrap_deadline"
    assert getattr(log_record, "timeout", None) == pytest.approx(selected_timeout)


def test_run_with_timeout_cancels_and_logs_stack(caplog):
    caplog.set_level(logging.ERROR)
    stop_event = threading.Event()
    worker_released = threading.Event()

    def hanging_worker() -> None:
        while not stop_event.is_set():
            time.sleep(0.01)
        worker_released.set()

    with pytest.raises(TimeoutError):
        preseed._run_with_timeout(
            hanging_worker,
            timeout=0.05,
            description="hanging_worker",
            cancel=stop_event.set,
        )

    assert stop_event.is_set()
    assert worker_released.wait(1.0)
    assert "stack trace for hanging_worker thread" in caplog.text
    assert "hanging_worker" in caplog.text
