import importlib


def test_call_remote_skips_during_boot_grace(monkeypatch):
    monkeypatch.setenv("VECTOR_SERVICE_URL", "http://example.com")
    monkeypatch.setenv("VECTOR_SERVICE_REMOTE_BOOT_GRACE", "60")

    import vector_service.vectorizer as vectorizer

    vectorizer = importlib.reload(vectorizer)
    monkeypatch.setattr(vectorizer, "_VECTOR_SERVICE_BOOT_TS", 0.0)
    monkeypatch.setattr(vectorizer.time, "monotonic", lambda: 10.0)

    called = {"count": 0}

    def fake_urlopen(*_args, **_kwargs):
        called["count"] += 1
        raise AssertionError("urlopen should not be called during boot grace")

    monkeypatch.setattr(vectorizer.urllib.request, "urlopen", fake_urlopen)

    svc = object.__new__(vectorizer.SharedVectorService)
    assert (
        vectorizer.SharedVectorService._call_remote(
            svc, "/vectorise", {"kind": "test", "record": {}}
        )
        is None
    )
    assert called["count"] == 0
