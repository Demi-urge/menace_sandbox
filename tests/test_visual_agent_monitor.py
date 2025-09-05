import time
from pathlib import Path
import types
from visual_agent_queue import VisualAgentQueue

import pytest


def _load_monitor():
    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"  # path-ignore
    text = path.read_text().splitlines()
    def _extract(name):
        start = next(i for i,l in enumerate(text) if l.startswith(name))
        indent = len(text[start]) - len(text[start].lstrip())
        end = start + 1
        while end < len(text) and (not text[end].strip() or text[end].startswith(" " * (indent+1)) or text[end].startswith(" "*indent)):
            end += 1
        return "\n".join(text[start:end])

    func_src = _extract("def _visual_agent_running")
    class_src = _extract("class VisualAgentMonitor")
    ns = {}
    exec(
        "import threading, os, sys, importlib, importlib.util\nfrom pathlib import Path\n_pkg_dir=Path('.')\nAGENT_MONITOR_INTERVAL=0.01\n"
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


def test_monitor_triggers_recover_on_health_failure(monkeypatch):
    manager = DummyManager()
    mon = VisualAgentMonitor(manager, "http://x", interval=0.01)
    monkeypatch.setitem(mon._loop.__globals__, "_visual_agent_running", lambda u: True)

    called = {}

    def fake_get(url, timeout=0):
        called[url] = called.get(url, 0) + 1
        class Resp:
            status_code = 500
        return Resp()

    def fake_post(url, headers=None, timeout=0):
        called[url] = called.get(url, 0) + 1
        class Resp:
            status_code = 200
            def json(self):
                return {}
        return Resp()

    import types, sys
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=fake_post, get=fake_get),
    )

    mon.start()
    try:
        for _ in range(50):
            if any(url.endswith("/recover") for url in called):
                break
            time.sleep(0.01)
    finally:
        mon.stop()

    assert any(url.endswith("/recover") for url in called)
