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
