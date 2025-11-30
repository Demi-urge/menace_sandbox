import types

import menace.error_bot as eb


def test_error_bot_reuses_broker_placeholder(monkeypatch):
    monkeypatch.setattr(eb, "_context_builder", None, raising=False)
    monkeypatch.setattr(eb, "_engine", None, raising=False)
    monkeypatch.setattr(eb, "_pipeline", None, raising=False)
    monkeypatch.setattr(eb, "_pipeline_promoter", None, raising=False)
    monkeypatch.setattr(eb, "_manager_instance", None, raising=False)
    monkeypatch.setattr(eb, "_evolution_orchestrator", None, raising=False)
    monkeypatch.setattr(eb, "_thresholds", None, raising=False)

    monkeypatch.setattr(eb, "create_context_builder", lambda: types.SimpleNamespace())
    monkeypatch.setattr(eb, "CodeDB", lambda: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(eb, "GPTMemoryManager", lambda: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(eb, "get_orchestrator", lambda *_, **__: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        eb,
        "get_thresholds",
        lambda *_: types.SimpleNamespace(roi_drop=1, error_increase=1, test_failure_increase=1),
        raising=False,
    )
    monkeypatch.setattr(eb, "persist_sc_thresholds", lambda *_, **__: None, raising=False)
    monkeypatch.setattr(eb, "ThresholdService", lambda: types.SimpleNamespace(), raising=False)

    monkeypatch.setattr(eb, "get_active_bootstrap_pipeline", lambda: (None, None))

    def _fail_prepare(*_, **__):  # pragma: no cover - should not be invoked
        raise AssertionError("prepare_pipeline_for_bootstrap should be bypassed")

    monkeypatch.setattr(eb, "prepare_pipeline_for_bootstrap", _fail_prepare)

    sentinel_manager = types.SimpleNamespace(bootstrap_placeholder=True)
    sentinel_pipeline = types.SimpleNamespace(manager=sentinel_manager, bootstrap_placeholder=True)

    class DummyBroker:
        def __init__(self):
            self.advertised: list[dict[str, object]] = []
            self.active_owner = True

        def resolve(self):
            return sentinel_pipeline, sentinel_manager

        def advertise(self, **kwargs):
            self.advertised.append(kwargs)

    broker = DummyBroker()
    monkeypatch.setattr(eb, "_bootstrap_dependency_broker", lambda: broker)

    manager = eb._ensure_self_coding_manager()

    assert manager is sentinel_manager
    assert eb._pipeline is sentinel_pipeline
    assert broker.advertised

