import types
import menace.dependency_watchdog as dw


def test_retry_backoff(monkeypatch):
    calls = []

    def fake_get(url, timeout=3):
        calls.append(url)
        if len(calls) < 3:
            raise Exception("boom")
        return types.SimpleNamespace(status_code=200)

    sleeps = []
    monkeypatch.setattr(dw, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(dw.time, "sleep", lambda s: sleeps.append(s))

    wd = dw.DependencyWatchdog({"svc": "u"}, {}, attempts=3, delay=1.0)
    wd.check()

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_auto_restart(monkeypatch):
    def fail_get(url, timeout=3):
        raise Exception("fail")

    monkeypatch.setattr(dw, "requests", types.SimpleNamespace(get=fail_get))
    monkeypatch.setattr(dw.time, "sleep", lambda s: None)

    called = []

    class DummyProv:
        def provision(self):
            called.append(True)

    monkeypatch.setattr(dw, "ExternalDependencyProvisioner", lambda: DummyProv())

    wd = dw.DependencyWatchdog({"svc": "u"}, {}, attempts=2, delay=0.0, auto_restart=True)
    wd.check()

    assert called == [True]
