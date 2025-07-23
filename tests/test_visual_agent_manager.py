import os
import socket
import sys
import time
import types
from pathlib import Path
import requests

from tests.test_visual_agent_subprocess_recovery import _write_script


def _write_script_psutil(path: Path) -> Path:
    script = _write_script(path)
    text = script.read_text()
    text = text.replace(
        "sys.argv = ['menace_visual_agent_2']",
        "sys.argv = ['menace_visual_agent_2']\nimport types, sys\npsutil = types.ModuleType('psutil'); psutil.pid_exists = lambda *_a, **_k: False; sys.modules['psutil'] = psutil",
    )
    script.write_text(text)
    return script
from visual_agent_manager import VisualAgentManager

TOKEN1 = "tok1"
TOKEN2 = "tok2"


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


def test_manager_rotates_token(tmp_path, monkeypatch):
    script = _write_script_psutil(tmp_path)
    port = _free_port()
    monkeypatch.setenv("MENACE_AGENT_PORT", str(port))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(tmp_path / "agent.lock"))
    monkeypatch.setenv("VISUAL_AGENT_PID_FILE", str(tmp_path / "agent.pid"))
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(root))

    mgr = VisualAgentManager(str(script))
    mgr.start(TOKEN1)
    _wait_status(port)

    h = {"x-token": TOKEN1}
    r1 = requests.post(
        f"http://127.0.0.1:{port}/run", json={"prompt": "p"}, headers=h, timeout=5
    )
    time.sleep(0.1)
    r2 = requests.post(
        f"http://127.0.0.1:{port}/run", json={"prompt": "p"}, headers=h, timeout=5
    )
    assert r1.status_code == 202
    assert r2.status_code == 409

    mgr.restart_with_token(TOKEN2)
    _wait_status(port)

    r_old = requests.post(
        f"http://127.0.0.1:{port}/run", json={"prompt": "p"}, headers={"x-token": TOKEN1}, timeout=5
    )
    assert r_old.status_code == 401
    r3 = requests.post(
        f"http://127.0.0.1:{port}/run", json={"prompt": "p"}, headers={"x-token": TOKEN2}, timeout=5
    )
    r4 = requests.post(
        f"http://127.0.0.1:{port}/run", json={"prompt": "p"}, headers={"x-token": TOKEN2}, timeout=5
    )
    assert r3.status_code == 202
    assert r4.status_code == 409

    mgr.shutdown()
