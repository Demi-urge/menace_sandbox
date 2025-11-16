import os
import sys
import json
import types
from pathlib import Path
import pytest
import sandbox_runner.environment as env

# Skip when hardware variable not set
if not os.environ.get("MENACE_HARDWARE"):
    pytest.skip("hardware not available", allow_module_level=True)


@pytest.mark.parametrize("os_type,image_key", [
    ("linux", "image"),
    ("windows", "windows_image"),
    ("macos", "macos_image"),
])
def test_full_environment_qemu(monkeypatch, tmp_path, os_type, image_key):
    tmp_clone = tmp_path / os_type
    vm_calls = []

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(tmp_clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore
    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/qemu-system-x86_64")

    def fake_run(cmd, **kwargs):
        vm_calls.append(cmd)
        if cmd[0] == "qemu-img":
            Path(cmd[-1]).touch()
        else:
            (tmp_clone / "data").mkdir(exist_ok=True)
            (tmp_clone / "data" / "roi_history.json").write_text("[1.0]")
        return types.SimpleNamespace()
    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.roi_history = []
            self.diagnostics = {}

        def load_history(self, path):
            self.loaded = path
            with open(path, "r") as f:
                self.roi_history = json.load(f)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({"OS_TYPE": os_type, "VM_SETTINGS": {image_key: "img.qcow2", "memory": "512M"}})
    assert Path(tmp_clone / "data" / "roi_history.json").exists()
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert tracker.roi_history == [1.0]
    assert not (tmp_clone / "overlay.qcow2").exists()
    if os_type in {"windows", "macos"}:
        assert any("qemu-system" in c[0] for c in vm_calls)
    else:
        assert vm_calls and vm_calls[0][0] == "python"
