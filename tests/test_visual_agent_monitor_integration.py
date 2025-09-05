import socket
import time
from pathlib import Path

import pytest
import requests

from tests.test_visual_agent_manager import _write_script_psutil
from visual_agent_manager import VisualAgentManager

TOKEN = "tok"


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
        "import threading, os, sys, importlib, importlib.util\nfrom pathlib import Path\n_pkg_dir=Path('.')\nAGENT_MONITOR_INTERVAL=0.2\n"
        + func_src
        + "\n"
        + class_src,
        ns,
    )
    return ns["VisualAgentMonitor"]


VisualAgentMonitor = _load_monitor()


def _free_port():
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_status(port):
    for _ in range(50):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/status", timeout=0.2)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def test_monitor_restarts_agent_and_recovers_queue(tmp_path, monkeypatch):
    script = _write_script_psutil(tmp_path)
    port = _free_port()
    monkeypatch.setenv("MENACE_AGENT_PORT", str(port))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(tmp_path / "agent.lock"))
    monkeypatch.setenv("VISUAL_AGENT_PID_FILE", str(tmp_path / "agent.pid"))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", TOKEN)
    monkeypatch.setenv("VISUAL_AGENT_AUTO_RECOVER", "1")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(root))

    mgr = VisualAgentManager(str(script))
    mon = VisualAgentMonitor(mgr, f"http://127.0.0.1:{port}", interval=0.2)

    mgr.start(TOKEN)
    _wait_status(port)

    resp = requests.post(
        f"http://127.0.0.1:{port}/run",
        headers={"x-token": TOKEN},
        json={"prompt": "p"},
        timeout=5,
    )
    assert resp.status_code == 202
    task_id = resp.json()["id"]

    pid_path = tmp_path / "agent.pid"
    old_pid = int(pid_path.read_text())
    mgr.process.terminate()
    try:
        mgr.process.wait(timeout=5)
    except Exception:
        mgr.process.kill()
        mgr.process.wait(timeout=5)

    class OneShot:
        def __init__(self):
            self.calls = 0
        def is_set(self):
            self.calls += 1
            return self.calls > 1
        def wait(self, t):
            pass
        def set(self):
            self.calls = 2

    mon._stop = OneShot()
    mon._loop()
    _wait_status(port)

    new_pid = mgr.process.pid
    assert new_pid != old_pid
    assert int(pid_path.read_text()) == new_pid

    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/status/{task_id}", timeout=1)
            if r.status_code == 200 and r.json().get("status") == "completed":
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        mon.stop()
        raise RuntimeError("task did not complete")

    mon.stop()
    assert not pid_path.exists()
