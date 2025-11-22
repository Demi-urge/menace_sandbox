import logging
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import sandbox.preseed_bootstrap as preseed


def test_prepare_pipeline_timeout_uses_deadline(monkeypatch, caplog):
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

    bootstrap_deadline = time.monotonic() + 50.0
    configured_timeout = 5.0
    preseed.initialize_bootstrap_context(
        bot_name="TestBot",
        use_cache=False,
        bootstrap_deadline=bootstrap_deadline,
        prepare_pipeline_timeout=configured_timeout,
    )

    selected_timeout = recorded_timeouts["prepare_pipeline_for_bootstrap"]
    assert selected_timeout > configured_timeout
    assert "prepare_pipeline_for_bootstrap timeout selected" in caplog.text
