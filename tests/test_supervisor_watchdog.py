import importlib.util
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Create minimal menace package to load module under test
spec_sw = importlib.util.spec_from_file_location(
    "menace.supervisor_watchdog",
    str(ROOT / "supervisor_watchdog.py"),  # path-ignore
)
sw = importlib.util.module_from_spec(spec_sw)
spec_sw.loader.exec_module(sw)
sys.modules["menace.supervisor_watchdog"] = sw


def test_restart_when_pid_missing(monkeypatch):
    monkeypatch.setattr(sw.os.path, "exists", lambda p: False)
    called = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        called["cmd"] = cmd
        class P:
            pass
        return P()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    wd = sw.SupervisorWatchdog(pid_file="x.pid", start_cmd=["python", "-m", "menace.service_supervisor"], check_interval=0)
    wd.check()
    assert called.get("cmd") == ["python", "-m", "menace.service_supervisor"]

def test_restart_when_process_dead(monkeypatch, tmp_path):
    pid_file = tmp_path / "sup.pid"
    pid_file.write_text("1234")
    monkeypatch.setattr(sw.os.path, "exists", lambda p: True)
    monkeypatch.setattr(sw.SupervisorWatchdog, "_pid_alive", lambda self, pid: False)
    called = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        called["cmd"] = cmd
        class P:
            pass
        return P()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    wd = sw.SupervisorWatchdog(pid_file=str(pid_file), start_cmd=["python", "-m", "menace.service_supervisor"], check_interval=0)
    wd.check()
    assert called.get("cmd") == ["python", "-m", "menace.service_supervisor"]
