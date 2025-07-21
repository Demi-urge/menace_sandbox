import os
import sys
import socket
import subprocess
import textwrap
import time
import threading
from pathlib import Path
import importlib.util
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")

# Ensure the menace package is importable
pkg_path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace", pkg_path / "__init__.py", submodule_search_locations=[str(pkg_path)]
)
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace_pkg
spec.loader.exec_module(menace_pkg)

from menace.visual_agent_client import VisualAgentClient


def _start_server(tmp_path: Path):
    script = tmp_path / "server.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, types, time, os
            heavy = ['cv2', 'numpy', 'mss']
            for name in heavy:
                sys.modules[name] = types.ModuleType(name)
            pyautogui = types.SimpleNamespace(
                moveTo=lambda *a, **k: None,
                click=lambda *a, **k: None,
                hotkey=lambda *a, **k: None,
                typewrite=lambda *a, **k: None,
                press=lambda *a, **k: None,
                write=lambda *a, **k: None,
            )
            sys.modules['pyautogui'] = pyautogui
            pt_mod = types.ModuleType('pytesseract')
            pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')
            pt_mod.image_to_string = lambda *a, **k: ''
            pt_mod.image_to_data = lambda *a, **k: {}
            pt_mod.Output = types.SimpleNamespace(DICT=0)
            sys.modules['pytesseract'] = pt_mod
            os.environ['VISUAL_AGENT_LOCK_FILE'] = os.path.join(
                os.environ.get('SANDBOX_DATA_DIR', '.'), 'server.lock')
            import menace_visual_agent_2 as va
            def fake_run(prompt: str, branch: str | None = None):
                time.sleep(0.3)
            va.run_menace_pipeline = fake_run
            import uvicorn
            uvicorn.run(
                va.app,
                host='127.0.0.1',
                port=int(os.environ['MENACE_AGENT_PORT']),
                log_level='error',
            )
            """
        )
    )
    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
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


def test_sequential_clients(tmp_path, monkeypatch):
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        os.environ["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "client.lock")
        client1 = VisualAgentClient(
            urls=[url], poll_interval=0.05, token="tombalolosvisualagent123"
        )
        client2 = VisualAgentClient(
            urls=[url], poll_interval=0.05, token="tombalolosvisualagent123"
        )

        def patched_poll(base: str):
            observed = False
            while True:
                resp = requests.get(f"{base}/status", timeout=1)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("active"):
                        observed = True
                    elif observed:
                        return True, "done"
                time.sleep(0.05)

        monkeypatch.setattr(client1, "_poll", patched_poll)
        monkeypatch.setattr(client2, "_poll", patched_poll)

        times = {}

        def run1():
            times["start1"] = time.time()
            client1.ask([{"content": "a"}])
            times["end1"] = time.time()

        def run2():
            time.sleep(0.01)
            times["start2"] = time.time()
            try:
                client2.ask([{"content": "b"}])
                times["error2"] = None
            except RuntimeError as exc:
                times["error2"] = str(exc)
            times["end2"] = time.time()

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        status = requests.get(f"{url}/status", timeout=1).json()
        assert status["queue"] == 0
        if times["error2"]:
            assert "409" in times["error2"]
        else:
            assert times["end2"] >= times["end1"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)
