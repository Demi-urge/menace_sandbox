from types import SimpleNamespace

import coding_bot_interface as cbi


def test_prepare_pipeline_recursion_reuses_broker(monkeypatch):
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()
    placeholder_sentinel = SimpleNamespace(bootstrap_placeholder=True)
    placeholder_pipeline = cbi._build_bootstrap_placeholder_pipeline(
        placeholder_sentinel
    )
    cbi.advertise_bootstrap_placeholder(
        dependency_broker=dependency_broker,
        pipeline=placeholder_pipeline,
        manager=placeholder_sentinel,
        owner=True,
    )
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: None)
    promise = cbi._BootstrapPipelinePromise()
    promise.resolve((placeholder_pipeline, lambda *_a, **_k: None))
    stub_coordinator = SimpleNamespace(
        peek_active=lambda: promise,
        claim=lambda: (True, promise),
        settle=lambda *_, **__: None,
    )
    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", stub_coordinator)

    def _fail_inner(**_kwargs: object) -> tuple[object, object]:
        raise AssertionError("prepare_pipeline_for_bootstrap_impl_inner should not run")

    monkeypatch.setattr(
        cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _fail_inner
    )
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 5, raising=False)

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=SimpleNamespace,
        context_builder=None,
        bot_registry=None,
        data_bot=None,
    )

    assert pipeline is placeholder_pipeline
    promote(None)
    assert dependency_broker.active_pipeline is placeholder_pipeline
    dependency_broker.clear()


def test_prepare_pipeline_nested_reuses_broker_and_suppresses_repeat_log(monkeypatch, caplog):
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()
    cbi._PREPARE_CALL_INVOCATIONS.clear()
    cbi._PREPARE_CALL_SUPPRESSIONS.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: None)

    placeholder_sentinel = SimpleNamespace(bootstrap_placeholder=True)
    placeholder_pipeline = cbi._build_bootstrap_placeholder_pipeline(
        placeholder_sentinel
    )

    call_count = 0

    def _stub_prepare(**_kwargs: object) -> tuple[object, object]:
        nonlocal call_count
        call_count += 1
        dependency_broker.advertise(
            pipeline=placeholder_pipeline,
            sentinel=placeholder_sentinel,
            owner=True,
        )
        return placeholder_pipeline, lambda *_a, **_k: None

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _stub_prepare)

    caplog.set_level("INFO")
    pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=SimpleNamespace,
        context_builder=None,
        bot_registry=None,
        data_bot=None,
    )
    promote(None)

    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)
    pipeline_reentry, promote_reentry = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=SimpleNamespace,
        context_builder=None,
        bot_registry=None,
        data_bot=None,
    )

    assert call_count == 1
    assert pipeline_reentry is placeholder_pipeline
    promote_reentry(None)
    dependency_broker.clear()
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 0, raising=False)

    prepare_invocations = [
        record
        for record in caplog.records
        if "calling prepare_pipeline_for_bootstrap" in record.message
    ]
    assert len(prepare_invocations) == 1
