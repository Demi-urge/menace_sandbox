import asyncio
import sys
import types
import json
from pathlib import Path
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
        self.removed = False
        self.stopped = False
        self.attrs = {"State": {"Health": {"Status": "healthy"}}}

    def reload(self):
        pass

    def stop(self, timeout=0):
        self.stopped = True

    def remove(self, force=True):
        self.removed = True

class DummyClient:
    def __init__(self):
        self.containers = type("C", (), {"run": lambda *a, **k: DummyContainer("new")})()


def _setup(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_METRICS.clear()
    env._STALE_CONTAINERS_REMOVED = 0
    env._STALE_VMS_REMOVED = 0
    env._RUNTIME_VMS_REMOVED = 0


def test_disk_usage_cleanup(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 999)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 999)
    monkeypatch.setattr(env, "_CONTAINER_DISK_LIMIT", 1)
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = env.time.time()
    env._CONTAINER_CREATED[c.id] = env.time.time()
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    monkeypatch.setattr(env, "_get_dir_usage", lambda p: 10)
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 0 and replaced == 1
    assert env._CLEANUP_METRICS["disk"] == 1
    assert env._STALE_CONTAINERS_REMOVED == 1
    metrics = asyncio.run(env.collect_metrics_async(0.0, 0.0, None))
    assert metrics.get("cleanup_disk") == 1.0
    assert metrics["stale_containers_removed"] == 1.0


def test_lifetime_cleanup(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 999)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 0.1)
    monkeypatch.setattr(env, "_CONTAINER_DISK_LIMIT", 0)
    times = [0.0]
    monkeypatch.setattr(env.time, "time", lambda: times[0])
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = 0.0
    env._CONTAINER_CREATED[c.id] = 0.0
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    times[0] = 0.2
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 0 and replaced == 1
    assert env._CLEANUP_METRICS["lifetime"] == 1
    assert env._STALE_CONTAINERS_REMOVED == 1
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["stale_containers_removed"] == 1.0


def test_get_dir_usage_error_logged(monkeypatch, tmp_path, caplog):
    path = tmp_path
    (path / "f.txt").write_text("data")

    def raise_error(_p):
        raise OSError("fail")

    monkeypatch.setattr(env.os.path, "getsize", raise_error)
    caplog.set_level("WARNING")
    assert env._get_dir_usage(str(path)) == 0
    assert f"size check failed for {str(path)}" in caplog.text


def test_qemu_invocation_and_cleanup(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "1")
    tmp_clone.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")

    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda n: "/usr/bin/qemu-system-x86_64")
    monkeypatch.setattr(env, "_docker_available", lambda: False)

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "qemu-img":
            Path(cmd[-1]).touch()
        if "qemu-system" in cmd[0]:
            (tmp_clone / "data" / "roi_history.json").write_text("[]")
        return types.SimpleNamespace()

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None

        def load_history(self, path):
            self.loaded = path

    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2", "memory": "1G", "timeout": 5}})

    assert any("qemu-system" in c[0][0] for c in calls)
    overlay = tmp_clone / "overlay.qcow2"
    assert not overlay.exists()
    qemu_call = [c for c in calls if "qemu-system" in c[0][0]][0]
    cmd, kwargs = qemu_call
    assert any("mount_tag=repo" in p for p in cmd)
    assert str(overlay) in " ".join(cmd)
    assert kwargs.get("timeout") == 5
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")


def test_purge_leftovers_metrics(monkeypatch, tmp_path):
    _setup(monkeypatch)
    file = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", file)
    file.write_text(json.dumps(["a"]))

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))

    env.purge_leftovers()
    assert env._STALE_CONTAINERS_REMOVED == 1
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["stale_containers_removed"] == 1.0

