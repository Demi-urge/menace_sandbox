import importlib

import pytest


def test_research_aggregator_waits_for_readiness(monkeypatch):
    module = importlib.import_module("research_aggregator_bot")

    readiness_calls: list[float | None] = []

    class _UnreadySignal:
        def await_ready(self, timeout=None):
            readiness_calls.append(timeout)
            raise TimeoutError("bootstrap readiness not yet satisfied")

    monkeypatch.setattr(module, "_BOOTSTRAP_READINESS", _UnreadySignal())

    def _fail_bootstrap(*args, **kwargs):
        raise AssertionError("bootstrap should not execute when readiness blocks")

    monkeypatch.setattr(module, "ensure_bootstrapped", _fail_bootstrap)

    with pytest.raises(RuntimeError):
        module.ResearchAggregatorBot()

    assert readiness_calls


def test_prediction_manager_defers_to_bootstrap_gate(monkeypatch):
    module = importlib.import_module("prediction_manager_bot")

    gate_calls: list[tuple] = []

    def _fail_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))
        raise TimeoutError("bootstrap gate busy")

    monkeypatch.setattr(module, "resolve_bootstrap_placeholders", _fail_gate)

    with pytest.raises(TimeoutError):
        module._bootstrap_placeholders()

    assert gate_calls


def test_memory_layer_respects_existing_readiness(monkeypatch, tmp_path):
    module = importlib.import_module("memory_bot")

    calls: list[dict] = []

    def _ready_bootstrap(**kwargs):
        calls.append(kwargs)
        return {"ready": True}

    monkeypatch.setattr(module, "ensure_bootstrapped", _ready_bootstrap)

    storage = module.VectorMemoryStorage(path=tmp_path / "memory.json.gz", embedder=None)
    assert storage.path.exists()
    assert calls == [{}]
