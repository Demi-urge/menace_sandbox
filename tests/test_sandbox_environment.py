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
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore

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
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(file) + ".lock"))
    file.write_text(json.dumps(["a"]))

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))

    env.purge_leftovers()
    assert env._STALE_CONTAINERS_REMOVED == 1
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["stale_containers_removed"] == 1.0


def test_metrics_exporter_updates(monkeypatch):
    _setup(monkeypatch)

    stub = types.ModuleType("metrics_exporter")

    class DummyGauge:
        def __init__(self):
            self.value = None
        def set(self, v):
            self.value = v

    class DummyLabelGauge:
        def __init__(self):
            self.values = {}
        def labels(self, worker):
            def set_val(v):
                self.values[worker] = v
            return types.SimpleNamespace(set=set_val)

    for name in (
        "cleanup_idle",
        "cleanup_unhealthy",
        "cleanup_lifetime",
        "cleanup_disk",
        "stale_containers_removed",
        "stale_vms_removed",
        "cleanup_failures",
        "force_kills",
        "runtime_vms_removed",
    ):
        setattr(stub, name, DummyGauge())
    stub.cleanup_duration_gauge = DummyLabelGauge()

    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    env._CLEANUP_METRICS.clear()
    env._CLEANUP_METRICS.update(
        {"idle": 1, "unhealthy": 2, "lifetime": 3, "disk": 4}
    )
    env._STALE_CONTAINERS_REMOVED = 5
    env._STALE_VMS_REMOVED = 6
    env._CLEANUP_FAILURES = 7
    env._FORCE_KILLS = 8
    env._RUNTIME_VMS_REMOVED = 9
    env._CLEANUP_DURATIONS["cleanup"] = 1.5
    env._CLEANUP_DURATIONS["reaper"] = 2.5

    env.collect_metrics(0.0, 0.0, None)

    assert stub.cleanup_idle.value == 1.0
    assert stub.cleanup_unhealthy.value == 2.0
    assert stub.cleanup_lifetime.value == 3.0
    assert stub.cleanup_disk.value == 4.0
    assert stub.stale_containers_removed.value == 5.0
    assert stub.stale_vms_removed.value == 6.0
    assert stub.cleanup_failures.value == 7.0
    assert stub.force_kills.value == 8.0
    assert stub.runtime_vms_removed.value == 9.0
    assert stub.cleanup_duration_gauge.values["cleanup"] == 1.5
    assert stub.cleanup_duration_gauge.values["reaper"] == 2.5


def test_collect_metrics_gauge_failure(monkeypatch):
    _setup(monkeypatch)

    stub = types.ModuleType("metrics_exporter")

    class BadGauge:
        def __init__(self):
            self.calls = 0
        def set(self, v):
            self.calls += 1
            raise ValueError("bad")

    class BadLabelGauge:
        def labels(self, worker):
            class Obj:
                def set(self, v):
                    raise ValueError("bad")
            return Obj()

    for name in (
        "cleanup_idle",
        "cleanup_unhealthy",
    ):
        setattr(stub, name, BadGauge())
    stub.cleanup_duration_gauge = BadLabelGauge()

    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    env._CLEANUP_METRICS["idle"] = 1
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["cleanup_idle"] == 1.0
    # calls attempted but errors suppressed
    assert stub.cleanup_idle.calls == 1


def test_collect_metrics_active_counts(monkeypatch, tmp_path):
    _setup(monkeypatch)

    containers = tmp_path / "containers.json"
    overlays = tmp_path / "overlays.json"
    containers.write_text(json.dumps(["c1", "c2"]))
    overlays.write_text(json.dumps(["ov"]))

    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", containers)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(containers) + ".lock"))
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))

    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["active_containers"] == 2.0
    assert metrics["active_overlays"] == 1.0


def test_vm_missing_falls_back_to_local(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    tmp_clone.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    monkeypatch.setattr(env, "_docker_available", lambda: False)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)

    overlays = tmp_path / "overlays.json"
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "python":
            (tmp_clone / "data").mkdir(exist_ok=True)
            (tmp_clone / "data" / "roi_history.json").write_text("[]")
        return types.SimpleNamespace()

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}

        def load_history(self, path):
            self.loaded = path

    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2"}})

    assert calls and calls[-1][0] == "python"
    assert not (tmp_clone / "overlay.qcow2").exists()
    assert env._read_active_overlays() == []
    assert tracker.diagnostics.get("vm_error") == "qemu missing"
    assert tracker.diagnostics.get("local_execution") == "vm"
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")


def test_vm_command_error_fallback(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    tmp_clone.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/qemu-system-x86_64")
    monkeypatch.setattr(env, "_docker_available", lambda: False)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)

    overlays = tmp_path / "overlays.json"
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "qemu-img":
            Path(cmd[-1]).touch()
            return types.SimpleNamespace()
        if "qemu-system" in cmd[0]:
            raise OSError("fail")
        if cmd[0] == "python":
            (tmp_clone / "data").mkdir(exist_ok=True)
            (tmp_clone / "data" / "roi_history.json").write_text("[]")
        return types.SimpleNamespace()

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}

        def load_history(self, path):
            self.loaded = path

    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2"}})

    assert any("qemu-img" in c[0] or "qemu-system" in c[0] for c in calls)
    assert calls[-1][0] == "python"
    assert not (tmp_clone / "overlay.qcow2").exists()
    assert env._read_active_overlays() == []
    assert tracker.diagnostics.get("local_execution") == "vm"
    assert tracker.diagnostics.get("vm_error")
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")



