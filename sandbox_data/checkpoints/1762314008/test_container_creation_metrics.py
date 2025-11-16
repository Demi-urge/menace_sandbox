import asyncio
import types
import sys
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import sandbox_runner.environment as env
import pytest


def test_failed_creation_metrics_and_alert(monkeypatch):
    class FailingContainers:
        def run(self, *a, **k):
            raise RuntimeError("boom")

    class DummyClient:
        def __init__(self):
            self.containers = FailingContainers()

    stub = types.ModuleType("metrics_exporter")

    class IncGauge:
        def __init__(self):
            self.labels_called = []
        def labels(self, image):
            def inc():
                self.labels_called.append(image)
            def set_val(v):
                self.labels_called.append((image, v))
            return types.SimpleNamespace(inc=inc, set=set_val)

    stub.container_creation_failures_total = IncGauge()
    stub.container_creation_alerts_total = IncGauge()
    stub.container_creation_seconds = IncGauge()
    stub.container_creation_success_total = IncGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    alerts = []
    monkeypatch.setattr(env, "dispatch_alert", lambda *a, **k: alerts.append((a, k)))

    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    env._CREATE_FAILURES.clear()
    env._CONSECUTIVE_CREATE_FAILURES.clear()

    with pytest.raises(RuntimeError):
        asyncio.run(env._create_pool_container("img"))

    assert stub.container_creation_failures_total.labels_called == ["img"]
    assert stub.container_creation_alerts_total.labels_called == ["img"]
    assert any(isinstance(v, tuple) for v in stub.container_creation_seconds.labels_called)
    assert alerts


def test_metrics_failure_handled(monkeypatch):
    class DummyContainers:
        def run(self, *a, **k):
            return types.SimpleNamespace(id="c1", status="running", attrs={"State": {"Health": {"Status": "healthy"}}})

    class DummyClient:
        def __init__(self):
            self.containers = DummyContainers()

    stub = types.ModuleType("metrics_exporter")

    class BadGauge:
        def labels(self, image=None, worker=None):
            class Obj:
                def inc(self, *a, **k):
                    raise ValueError("boom")

                def set(self, *a, **k):
                    raise ValueError("boom")

            return Obj()

    stub.container_creation_success_total = BadGauge()
    stub.container_creation_seconds = BadGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)

    result = asyncio.run(env._create_pool_container("img"))
    assert result[0].id == "c1"
