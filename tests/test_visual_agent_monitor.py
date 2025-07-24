import time
from pathlib import Path

import pytest


def _load_monitor():
    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"
    lines = path.read_text().splitlines()
    func_src = "\n".join(lines[103:114])
    class_src = "\n".join(lines[222:254])
    ns = {}
    exec("import threading, os\nAGENT_MONITOR_INTERVAL=0.01\n" + func_src + "\n" + class_src, ns)
    return ns["VisualAgentMonitor"]


VisualAgentMonitor = _load_monitor()


class DummyManager:
    def __init__(self):
        self.calls = 0

    def restart_with_token(self, token):
        self.calls += 1

    def shutdown(self):
        pass


def test_monitor_restarts_when_status_fails(monkeypatch):
    manager = DummyManager()
    mon = VisualAgentMonitor(manager, "http://x", interval=0.01)
    monkeypatch.setitem(mon._loop.__globals__, "_visual_agent_running", lambda u: False)
    mon.start()
    try:
        for _ in range(20):
            if manager.calls:
                break
            time.sleep(0.01)
    finally:
        mon.stop()
    assert manager.calls > 0
