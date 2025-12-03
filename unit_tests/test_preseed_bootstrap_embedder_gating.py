from unit_tests.test_preseed_bootstrap_timeout_safety import (
    _load_preseed_bootstrap_module,
)


def _reset_embedder_state(preseed_bootstrap):
    preseed_bootstrap._BOOTSTRAP_CACHE.clear()
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None
    preseed_bootstrap._BOOTSTRAP_SCHEDULER.clear_embedder_deferral()


def test_embedder_skipped_without_budget(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()
    _reset_embedder_state(preseed_bootstrap)

    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_budget", lambda *_, **__: None
    )
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_deadline", lambda *_, **__: None
    )

    def fail_embedder(*_, **__):
        raise AssertionError("embedder preload should be skipped")

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", fail_embedder)

    preseed_bootstrap.initialize_bootstrap_context(
        bot_name="EmbedderSkipBudget", use_cache=False
    )

    deferred, reason = preseed_bootstrap._BOOTSTRAP_SCHEDULER.embedder_deferral()
    assert deferred is True
    assert reason == "embedder_preload_no_budget_window"
    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("deferral_reason") == "embedder_preload_no_budget_window"


def test_embedder_skipped_when_warmup_lite_forced(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()
    _reset_embedder_state(preseed_bootstrap)

    monkeypatch.setenv("MENACE_BOOTSTRAP_MODE", "lite")
    monkeypatch.setattr(preseed_bootstrap._StagedBootstrapController, "stage_budget", lambda *_, **__: 15.0)
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_deadline", lambda *_, **__: None
    )

    def fail_embedder(*_, **__):
        raise AssertionError("embedder preload should be skipped")

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", fail_embedder)

    preseed_bootstrap.initialize_bootstrap_context(
        bot_name="EmbedderSkipWarmupLite", use_cache=False
    )

    deferred, reason = preseed_bootstrap._BOOTSTRAP_SCHEDULER.embedder_deferral()
    assert deferred is True
    assert reason == "embedder_warmup_lite_forced"
    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("result") == preseed_bootstrap._BOOTSTRAP_PLACEHOLDER
    assert job.get("deferral_reason") == "embedder_warmup_lite_forced"

