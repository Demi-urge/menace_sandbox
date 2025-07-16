import os
import sys
import types
from pathlib import Path
import sandbox_runner.environment as env


def test_simulate_full_environment_vm(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    vm_calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")
    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda path, ignore_errors=True: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/qemu-system-x86_64")

    def fake_run(cmd, **kwargs):
        vm_calls.append(cmd)
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
    assert vm_calls and "qemu-system-x86_64" in vm_calls[0][0]
    assert any("img.qcow2" in part for part in vm_calls[0])
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")


def test_simulate_full_environment_vm_fallback(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone2"
    calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")

    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
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

