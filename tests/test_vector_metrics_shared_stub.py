import vector_metrics_db


def test_shared_db_stub_defers_activation(monkeypatch):
    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False
            self.weights = {}

        def set_db_weights(self, weights):
            self.weights.update(weights)

        def get_db_weights(self, default=None):
            if self.weights:
                return dict(self.weights)
            return default or {}

        def log_embedding(self, *args, **kwargs):
            calls.append(("log_embedding", args, kwargs))

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(bootstrap_fast=True)

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    vector_metrics_db.ensure_vector_db_weights(
        ["alpha", "beta"], bootstrap_fast=True, warmup=True
    )

    assert calls == []

    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert vector_metrics_db._VECTOR_DB_INSTANCE is vm
    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, DummyVectorDB)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is activated
    assert calls[0][0] == "init"
    activated.log_embedding("default", tokens=1, wall_time_ms=1.0)
    assert calls[1][0] == "log_embedding"
    assert getattr(activated, "weights", {}) == {"alpha": 1.0, "beta": 1.0}
