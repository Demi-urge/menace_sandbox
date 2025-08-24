import asyncio
import types
import pytest

from menace.trial_monitor import TrialMonitor, TrialConfig

class DummyDeployer:
    db = types.SimpleNamespace(trials=lambda status: [])
    bot_db = types.SimpleNamespace(
        conn=types.SimpleNamespace(execute=lambda *a, **k: [("bot",)]),
        router=types.SimpleNamespace(menace_id="test"),
    )

class DummyOptimizer:
    pass

class DummyDataBot:
    db = types.SimpleNamespace(fetch=lambda limit: [])


def _stop_after_first(mon: TrialMonitor):
    async def inner(_: float) -> None:
        mon.running = False
        raise SystemExit
    return inner


def test_loop_logs_exception(monkeypatch, caplog):
    mon = TrialMonitor(
        DummyDeployer(),
        DummyOptimizer(),
        DummyDataBot(),
        history_db=types.SimpleNamespace(),
        config=TrialConfig(interval=0),
    )
    mon.running = True
    monkeypatch.setattr(mon, "check_trials", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("menace.trial_monitor.asyncio.sleep", _stop_after_first(mon))
    caplog.set_level("ERROR")
    with pytest.raises(SystemExit):
        asyncio.run(mon._loop())
    assert "trial check failed" in caplog.text
    assert mon.failure_count == 1
