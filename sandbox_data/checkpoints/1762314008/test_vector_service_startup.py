import pytest


def test_ensure_vector_service_retries(monkeypatch):
    import vector_service.context_builder as cb

    calls = {"count": 0}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=2):
        calls["count"] += 1
        if calls["count"] < 3:
            raise OSError("not ready")
        return DummyResponse()

    import urllib.request
    import subprocess

    monkeypatch.setenv("VECTOR_SERVICE_URL", "http://example")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    popen_called = {"called": False}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_called.__setitem__("called", True))
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)

    cb._ensure_vector_service()
    assert calls["count"] == 3
    assert not popen_called["called"]


def test_ensure_vector_service_raises_after_attempts(monkeypatch):
    import vector_service.context_builder as cb

    calls = {"count": 0}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=2):
        calls["count"] += 1
        raise OSError("not ready")

    import urllib.request
    import subprocess

    monkeypatch.setenv("VECTOR_SERVICE_URL", "http://example")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    popen_called = {"count": 0}
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: popen_called.__setitem__("count", popen_called["count"] + 1),
    )
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)

    with pytest.raises(cb.VectorServiceError):
        cb._ensure_vector_service()
    assert popen_called["count"] == 1
    assert calls["count"] == 10
