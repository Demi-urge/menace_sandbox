import asyncio
import sandbox_runner.environment as env


def test_schedule_coroutine_closed_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(env, "_BACKGROUND_LOOP", loop, raising=False)
    monkeypatch.setattr(env, "_BACKGROUND_THREAD", None, raising=False)

    async def dummy():
        pass

    result = env._schedule_coroutine(dummy())
    assert result is None


def test_ensure_cleanup_worker_closed_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(env, "_BACKGROUND_LOOP", loop, raising=False)
    monkeypatch.setattr(env, "_BACKGROUND_THREAD", None, raising=False)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    monkeypatch.setattr(env, "start_container_event_listener", lambda: None)

    executed = []

    async def fake_cleanup():
        executed.append("cleanup")

    async def fake_reaper():
        executed.append("reaper")

    monkeypatch.setattr(env, "_cleanup_worker", fake_cleanup)
    monkeypatch.setattr(env, "_reaper_worker", fake_reaper)

    env._CLEANUP_TASK = None
    env._REAPER_TASK = None

    env.ensure_cleanup_worker()

    assert executed == ["cleanup", "reaper"]
    assert env._CLEANUP_TASK is None
    assert env._REAPER_TASK is None

