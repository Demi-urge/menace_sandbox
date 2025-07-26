import os
import sys
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tests.test_visual_agent_subprocess_recovery import _write_script, TOKEN

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")


def _start_server(tmp_path: Path):
    script = _write_script(tmp_path)
    text = script.read_text()
    text = text.replace(
        "sys.argv = ['menace_visual_agent_2']",
        "sys.argv = ['menace_visual_agent_2']\nimport types, sys\npsutil = types.ModuleType('psutil'); psutil.pid_exists = lambda *_a, **_k: False; sys.modules['psutil'] = psutil",
    )
    script.write_text(text)

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "agent.lock")
    env["VISUAL_AGENT_PID_FILE"] = str(tmp_path / "agent.pid")
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    for _ in range(50):
        try:
            requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("server did not start")

    return proc, port


def test_busy_run_shows_queue_length(tmp_path):
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        resp1 = requests.post(
            f"{url}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        resp2 = requests.post(
            f"{url}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp1.status_code == 202
        assert resp2.status_code == 202

        for _ in range(20):
            status = requests.get(f"{url}/status", timeout=1).json()
            if status.get("active"):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("agent did not start running")

        assert status["queue"] == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
