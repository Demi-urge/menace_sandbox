import runpy
import sys
import types
import pytest


def test_cleanup_removes_lock(tmp_path, monkeypatch):
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")
    class DummyTimeout(Exception):
        pass

    class DummyFileLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, timeout=0):
            pass

        def release(self):
            pass

    filelock_mod.FileLock = DummyFileLock
    filelock_mod.Timeout = DummyTimeout
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    psutil_mod = types.ModuleType("psutil")
    psutil_mod.pid_exists = lambda *_a, **_k: False
    monkeypatch.setitem(sys.modules, "psutil", psutil_mod)

    lock_path = tmp_path / "agent.lock"
    lock_path.write_text("")
    pid_path = tmp_path / "agent.pid"
    pid_path.write_text("123")

    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(lock_path))
    monkeypatch.setenv("VISUAL_AGENT_PID_FILE", str(pid_path))
    monkeypatch.setenv("VISUAL_AGENT_LOCK_TIMEOUT", "9999999")
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tombalolosvisualagent123")
    monkeypatch.setattr(sys, "argv", ["menace_visual_agent_2", "--cleanup"])

    with pytest.raises(SystemExit):
        runpy.run_module("menace_visual_agent_2", run_name="__main__")

    assert not lock_path.exists()
    assert not pid_path.exists()
