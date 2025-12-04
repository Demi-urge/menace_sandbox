from unit_tests.test_preseed_bootstrap_timeout_safety import (
    _load_preseed_bootstrap_module,
)


def _reset_embedder_state(preseed_bootstrap):
    preseed_bootstrap._BOOTSTRAP_CACHE.clear()
    preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB = None
    preseed_bootstrap._BOOTSTRAP_SCHEDULER.clear_embedder_deferral()
    if hasattr(preseed_bootstrap._BOOTSTRAP_SCHEDULER, "stage_controller"):
        preseed_bootstrap._BOOTSTRAP_SCHEDULER.stage_controller = None


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


def test_heavy_embedder_hints_ignored_in_bootstrap_lite(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()
    _reset_embedder_state(preseed_bootstrap)

    monkeypatch.setenv("MENACE_BOOTSTRAP_MODE", "lite")
    monkeypatch.setenv("BOOTSTRAP_HEAVY_BOOTSTRAP", "1")
    monkeypatch.setenv("MENACE_EMBEDDER_FULL_PRELOAD", "1")
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_budget", lambda *_, **__: 30.0
    )
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_deadline", lambda *_, **__: None
    )
    monkeypatch.setattr(
        preseed_bootstrap,
        "stage_for_step",
        lambda step: "vector_seeding" if step == "embedder_preload" else None,
    )

    def fail_embedder(*_, **__):
        raise AssertionError("embedder preload should be skipped in bootstrap-lite")

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", fail_embedder)

    preseed_bootstrap.initialize_bootstrap_context(
        bot_name="EmbedderSkipBootstrapLite", use_cache=False
    )

    deferred, reason = preseed_bootstrap._BOOTSTRAP_SCHEDULER.embedder_deferral()
    assert deferred is True
    assert reason == "embedder_preload_bootstrap_lite"
    job = preseed_bootstrap._BOOTSTRAP_EMBEDDER_JOB or {}
    assert job.get("deferral_reason") == "embedder_preload_bootstrap_lite"
    controller = getattr(preseed_bootstrap._BOOTSTRAP_SCHEDULER, "stage_controller", None)
    assert controller is not None
    assert controller.deferred_reason(stage="vector_seeding") == "embedder_preload_bootstrap_lite"


def test_force_embedder_preload_overrides_bootstrap_lite_guard(monkeypatch):
    preseed_bootstrap = _load_preseed_bootstrap_module()
    _reset_embedder_state(preseed_bootstrap)

    monkeypatch.setenv("MENACE_BOOTSTRAP_MODE", "lite")
    monkeypatch.setenv("MENACE_EMBEDDER_FULL_PRELOAD", "1")
    monkeypatch.setenv("MENACE_FORCE_EMBEDDER_PRELOAD", "1")
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_budget", lambda *_, **__: 30.0
    )
    monkeypatch.setattr(
        preseed_bootstrap._StagedBootstrapController, "stage_deadline", lambda *_, **__: None
    )
    monkeypatch.setattr(
        preseed_bootstrap,
        "stage_for_step",
        lambda step: "vector_seeding" if step == "embedder_preload" else None,
    )

    calls = {"embedder": 0}

    def fake_embedder(*_, **__):
        calls["embedder"] += 1
        return {"embedder": "loaded"}

    monkeypatch.setattr(preseed_bootstrap, "_bootstrap_embedder", fake_embedder)

    preseed_bootstrap.initialize_bootstrap_context(
        bot_name="EmbedderForceBootstrapLite", use_cache=False
    )

    assert calls["embedder"] == 1
    deferred, reason = preseed_bootstrap._BOOTSTRAP_SCHEDULER.embedder_deferral()
    assert deferred is False
    assert reason is None
    controller = getattr(preseed_bootstrap._BOOTSTRAP_SCHEDULER, "stage_controller", None)
    assert controller is not None
    assert controller.deferred_reason(stage="vector_seeding") is None

