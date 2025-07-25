import types
import sandbox_runner.environment as env

def setup_container(monkeypatch):
    c = types.SimpleNamespace(id="c1")
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = "dir"
    env._CONTAINER_LAST_USED[c.id] = 0.0
    env._CONTAINER_CREATED[c.id] = 0.0
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(env.time, "time", lambda: 1.0)
    monkeypatch.setattr(env, "_check_disk_usage", lambda cid: False)
    monkeypatch.setattr(env, "_verify_container", lambda c: True)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    return c


def test_cleanup_logging_success(monkeypatch):
    entries = []
    monkeypatch.setattr(env, "_write_cleanup_log", lambda e: entries.append(e))
    c = setup_container(monkeypatch)
    monkeypatch.setattr(env, "_stop_and_remove", lambda cont: True)
    env._cleanup_idle_containers()
    assert entries and entries[0]["resource_id"] == c.id
    assert entries[0]["reason"] == "idle"
    assert entries[0]["success"] is True


def test_cleanup_logging_failure(monkeypatch):
    entries = []
    monkeypatch.setattr(env, "_write_cleanup_log", lambda e: entries.append(e))
    c = setup_container(monkeypatch)
    monkeypatch.setattr(env, "_stop_and_remove", lambda cont: False)
    env._cleanup_idle_containers()
    assert entries and entries[0]["resource_id"] == c.id
    assert entries[0]["reason"] == "idle"
    assert entries[0]["success"] is False
