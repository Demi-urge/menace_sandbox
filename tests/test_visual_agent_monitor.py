import time
from pathlib import Path
import types
from visual_agent_queue import VisualAgentQueue

import pytest


def _load_monitor():
    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"
    lines = path.read_text().splitlines()
    func_src = "\n".join(lines[106:116])
    class_src = "\n".join(lines[222:275])
    ns = {}
    exec(
        "import threading, os\nfrom pathlib import Path\nAGENT_MONITOR_INTERVAL=0.01\n"
        + func_src
        + "\n"
        + class_src,
        ns,
    )
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


def test_monitor_checks_integrity(monkeypatch, tmp_path):
    manager = DummyManager()
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    queue_db = tmp_path / "visual_agent_queue.db"
    queue_db.write_text("bad")

    mon = VisualAgentMonitor(manager, "http://x", interval=0.01)
    monkeypatch.setitem(mon._loop.__globals__, "_visual_agent_running", lambda u: True)

    called = {}

    def fake_post(url, headers=None, timeout=0):
        called[url] = called.get(url, 0) + 1
        if url.endswith("/integrity"):
            q = VisualAgentQueue(queue_db)
            rebuilt = q.check_integrity()
            class Resp:
                status_code = 200
                def json(self):
                    return {"rebuilt": rebuilt}
            return Resp()
        class Resp:
            status_code = 200
            def json(self):
                return {}
        return Resp()

    import types, sys
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))

    mon.start()
    try:
        for _ in range(50):
            if any(url.endswith("/integrity") for url in called):
                break
            time.sleep(0.01)
    finally:
        mon.stop()

    assert any(url.endswith("/integrity") for url in called)
    backups = list(tmp_path.glob("visual_agent_queue.db.corrupt.*"))
    assert backups
