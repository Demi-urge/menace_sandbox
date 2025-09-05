import os
import sys
import types
import subprocess
from pathlib import Path
import sandbox_runner.environment as env


def test_simulate_full_environment_vm(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    vm_calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    tmp_clone.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    overlays = tmp_path / "overlays.json"
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore
    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda path, ignore_errors=True: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/qemu-system-x86_64")

    def fake_run(cmd, **kwargs):
        vm_calls.append(cmd)
        if cmd[0] == "qemu-img":
            Path(cmd[-1]).touch()
        else:
            (tmp_clone / "data").mkdir(exist_ok=True)
            (tmp_clone / "data" / "roi_history.json").write_text("[]")
        return types.SimpleNamespace()
    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
        def load_history(self, path):
            self.loaded = path
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2", "memory": "1G"}})
    assert vm_calls and any("qemu-system" in c[0] for c in vm_calls)
    qemu_cmd = next(c for c in vm_calls if "qemu-system" in c[0])
    assert any("overlay.qcow2" in part for part in qemu_cmd)
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")


def test_simulate_full_environment_vm_fallback(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone2"
    calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore

    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
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

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2", "memory": "1G"}})
    assert calls and calls[0][0] == "python"
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert getattr(tracker, "diagnostics", {}).get("local_execution") == "vm"


def test_simulate_full_environment_vm_timeout(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone_timeout"
    tmp_clone.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore

    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/qemu-system-x86_64")

    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "qemu-img":
            Path(cmd[-1]).touch()
            return types.SimpleNamespace()
        if "qemu-system" in cmd[0]:
            raise subprocess.TimeoutExpired(cmd[0], kwargs.get("timeout", 1))
        (tmp_clone / "data").mkdir(exist_ok=True)
        (tmp_clone / "data" / "roi_history.json").write_text("[]")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}

        def load_history(self, path):
            self.loaded = path

    monkeypatch.setitem(
        sys.modules,
        "menace.roi_tracker",
        types.SimpleNamespace(ROITracker=DummyTracker),
    )

    tracker = env.simulate_full_environment({"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": "img.qcow2"}})

    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert tracker.diagnostics.get("vm_error") == "timeout"
    assert tracker.diagnostics.get("local_execution") == "vm"
    assert not tmp_clone.exists()


def test_vm_overlay_cleanup_backoff(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "overlay_backoff"
    tmp_clone.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore

    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "which", lambda n: "/usr/bin/qemu-system-x86_64")

    delays = []
    monkeypatch.setattr(env.time, "sleep", lambda d: delays.append(d))
    monkeypatch.setattr(env, "_CREATE_BACKOFF_BASE", 1.0)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 3)

    def fake_run(cmd, **kwargs):
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

    assert delays == [1.0, 2.0]
    assert not (tmp_clone / "overlay.qcow2").exists()
    assert env._read_active_overlays() == []
    assert tracker.diagnostics.get("local_execution") == "vm"
    assert tracker.diagnostics.get("vm_error")
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")

