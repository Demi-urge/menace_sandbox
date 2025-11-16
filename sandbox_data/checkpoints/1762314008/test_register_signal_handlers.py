import signal
import sandbox_runner.environment as env


def test_register_signal_handlers(monkeypatch):
    calls = []
    handlers = []
    def fake_signal(sig, handler):
        calls.append(sig)
        handlers.append(handler)
    monkeypatch.setattr(env.signal, "signal", fake_signal)

    called = []
    monkeypatch.setattr(env, "_cleanup_pools", lambda: called.append("cleanup"))
    monkeypatch.setattr(env, "_await_cleanup_task", lambda: called.append("await"))
    monkeypatch.setattr(env, "_await_reaper_task", lambda: called.append("reaper"))

    env.register_signal_handlers()

    assert signal.SIGTERM in calls and signal.SIGINT in calls
    assert len(handlers) == 2

    for h in handlers:
        h(None, None)

    assert called == ["cleanup", "await", "reaper", "cleanup", "await", "reaper"]
