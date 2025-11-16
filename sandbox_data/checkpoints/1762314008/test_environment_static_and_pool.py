import asyncio
import types
import os
import sandbox_runner.environment as env


class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
        self.stopped = False
        self.removed = False

    def reload(self):
        pass

    def stop(self, timeout=0):
        self.stopped = True

    def remove(self, force=True):
        self.removed = True


async def _fake_create(image: str):
    c = DummyContainer("x")
    td = os.path.join("/tmp", c.id)
    return c, td


def test_static_behavior_analysis_detection(monkeypatch):
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    code = """
import os
import requests
import subprocess

def run():
    subprocess.Popen(['ls'])
    os.system('sudo echo hi')
    requests.get('http://example.com')
    open('out.txt', 'w')
    eval('1+1')
"""
    res = env.static_behavior_analysis(code)
    flags = res["flags"]
    assert "process call subprocess.Popen" in flags
    assert "privilege escalation sudo" in flags
    assert "network call requests.get" in flags
    assert "dangerous call eval" in flags
    assert "file write" in flags
    assert "import dangerous module subprocess" in flags
    assert "import dangerous module requests" in flags
    assert res["files_written"] == ["out.txt"]
    assert res.get("regex_flags") == ["raw_dangerous_pattern"]



def test_pooled_container_reuse_and_cleanup(monkeypatch, tmp_path):
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    created = []

    async def fake_create(image: str):
        c = DummyContainer(len(created))
        td = tmp_path / c.id
        td.mkdir()
        created.append(c)
        env._CONTAINER_DIRS[c.id] = str(td)
        env._CONTAINER_LAST_USED[c.id] = env.time.time()
        env._CONTAINER_CREATED[c.id] = env.time.time()
        return c, str(td)

    monkeypatch.setattr(env, "_create_pool_container", fake_create)

    c1, td1 = asyncio.run(env._get_pooled_container("img"))
    env._release_container("img", c1)
    assert created == [c1]

    c2, td2 = asyncio.run(env._get_pooled_container("img"))
    assert c2 is c1
    assert td2 == td1
    env._release_container("img", c2)

    env._cleanup_pools()
    assert not env._CONTAINER_POOLS
    assert c1.stopped and c1.removed
    assert not (tmp_path / c1.id).exists()


def test_release_non_running_container(monkeypatch, tmp_path):
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    c = DummyContainer("bad")
    c.status = "exited"
    td = tmp_path / c.id
    td.mkdir()
    env._CONTAINER_DIRS[c.id] = str(td)
    env._release_container("img", c)
    assert not env._CONTAINER_POOLS.get("img")
    assert c.removed
    assert not td.exists()
