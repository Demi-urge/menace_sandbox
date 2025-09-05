import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "visual_agent_watchdog",
    str(ROOT / "visual_agent_watchdog.py"),  # path-ignore
)
va_wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(va_wd)
sys.modules["visual_agent_watchdog"] = va_wd


class DummyManager:
    def __init__(self, pid_file: Path):
        self.pid_file = pid_file
        self.calls = 0

    def restart_with_token(self, token: str):
        self.calls += 1


def test_restart_when_pid_missing(tmp_path, monkeypatch):
    pid_file = tmp_path / "agent.pid"
    mgr = DummyManager(pid_file)
    wd = va_wd.VisualAgentWatchdog(mgr, check_interval=0)
    wd.check()
    assert mgr.calls == 1


def test_restart_when_process_dead(tmp_path, monkeypatch):
    pid_file = tmp_path / "agent.pid"
    pid_file.write_text("123")
    mgr = DummyManager(pid_file)
    monkeypatch.setattr(va_wd, "_pid_alive", lambda pid: False)
    wd = va_wd.VisualAgentWatchdog(mgr, check_interval=0)
    wd.check()
    assert mgr.calls == 1


def test_no_restart_when_alive(tmp_path, monkeypatch):
    pid_file = tmp_path / "agent.pid"
    pid_file.write_text("123")
    mgr = DummyManager(pid_file)
    monkeypatch.setattr(va_wd, "_pid_alive", lambda pid: True)
    wd = va_wd.VisualAgentWatchdog(mgr, check_interval=0)
    wd.check()
    assert mgr.calls == 0
