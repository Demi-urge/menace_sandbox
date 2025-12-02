import importlib


class _StubVectorMetricsDB:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.activated = False

    def activate_on_first_write(self):
        self.activated = True


def _reload_module(monkeypatch):
    import embeddable_db_mixin as edm

    importlib.reload(edm)
    monkeypatch.setattr(edm, "_VEC_METRICS", None)
    return edm


def _fake_resolve(*, bootstrap_fast=None, warmup=None):
    return (bool(bootstrap_fast), bool(warmup), False, False)


def test_vector_metrics_stub_activates_on_callsite_warmup(monkeypatch):
    edm = _reload_module(monkeypatch)

    monkeypatch.setattr(edm, "resolve_vector_bootstrap_flags", _fake_resolve)
    monkeypatch.setattr(
        edm, "get_bootstrap_vector_metrics_db", lambda **kwargs: _StubVectorMetricsDB(**kwargs)
    )

    vm = edm._vector_metrics_db(warmup=True)

    assert isinstance(vm, _StubVectorMetricsDB)
    assert vm.kwargs["bootstrap_fast"] is False
    assert vm.kwargs["warmup"] is True
    assert vm.kwargs["ensure_exists"] is False
    assert vm.kwargs["read_only"] is True
    assert vm.activated is True


def test_vector_metrics_stub_activates_on_warmup_lite(monkeypatch):
    edm = _reload_module(monkeypatch)

    monkeypatch.setattr(edm, "resolve_vector_bootstrap_flags", _fake_resolve)
    monkeypatch.setattr(
        edm, "get_bootstrap_vector_metrics_db", lambda **kwargs: _StubVectorMetricsDB(**kwargs)
    )

    vm = edm._vector_metrics_db(warmup=False, warmup_lite=True)

    assert isinstance(vm, _StubVectorMetricsDB)
    assert vm.kwargs["bootstrap_fast"] is False
    assert vm.kwargs["warmup"] is True
    assert vm.kwargs["ensure_exists"] is False
    assert vm.kwargs["read_only"] is True
    assert vm.activated is True
