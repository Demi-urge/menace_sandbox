import os
import sys
import socket
import subprocess
import textwrap
import time
import threading
from pathlib import Path
import importlib.util
import argparse
import json
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")
from fastapi.testclient import TestClient

TOKEN = "tombalolosvisualagent123"

# Ensure the menace package is importable
pkg_path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace", pkg_path / "__init__.py", submodule_search_locations=[str(pkg_path)]  # path-ignore
)
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace_pkg
spec.loader.exec_module(menace_pkg)

from menace.visual_agent_client import VisualAgentClient


def _start_server(tmp_path: Path):
    script = tmp_path / "server.py"  # path-ignore
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
            pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
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
        assert times["error2"] is None
        assert times["end2"] >= times["end1"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_queue_runs_sequentially(tmp_path, monkeypatch):
    """Tasks enqueued via ``ask_async`` should run one after another."""

    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(tmp_path / "lock"))
    vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))

    active = 0
    overlap = []

    def fake_send(self, base, prompt):
        nonlocal active, overlap
        # lock file should exist while the task executes
        assert os.path.exists(vac_mod.GLOBAL_LOCK_PATH)
        active += 1
        if active > 1:
            overlap.append(True)
        time.sleep(0.05)
        active -= 1
        return True, "ok"

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_send", fake_send)
    client = vac_mod.VisualAgentClient(urls=["http://x"], poll_interval=0.01)

    futs = [client.ask_async([{"content": str(i)}]) for i in range(3)]
    results = [f.result(timeout=2) for f in futs]

    assert all(r["choices"][0]["message"]["content"] == "ok" for r in results)
    assert not overlap
    assert not os.path.exists(vac_mod.GLOBAL_LOCK_PATH)


def test_full_autonomous_updates_histories(monkeypatch, tmp_path):
    """ROI and synergy histories should grow during autonomous runs."""

    import sandbox_runner.cli as cli
    dummy_preset = {"env": "dev"}
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [dummy_preset])

    class DummyTracker:
        def __init__(self):
            self.module_deltas = {"m": [0.1]}
            self.metrics_history = {"synergy_roi": [0.2]}
            self.roi_history = [0.1]
            self.synergy_history = [{"synergy_roi": 0.2}]
            self.synergy_metrics_history = {}
            self.predicted_roi = []
            self.actual_roi = []
            self.predicted_metrics = {}
            self.actual_metrics = {}
            self.scenario_synergy = {}

        def save_history(self, path):
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                        "synergy_history": self.synergy_history,
                    }
                )
            )

        def diminishing(self):
            return 0.01

        def rankings(self):
            return [("m", 0.1, 0.1)]

    def fake_capture(preset, args):
        t = DummyTracker()
        Path(args.sandbox_data_dir).mkdir(parents=True, exist_ok=True)
        t.save_history(str(Path(args.sandbox_data_dir) / "roi_history.json"))
        return t

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    hist: list[dict[str, float]] = []
    ma_hist: list[dict[str, float]] = []

    args = argparse.Namespace(
        sandbox_data_dir=str(tmp_path),
        preset_count=1,
        max_iterations=1,
        dashboard_port=None,
        roi_cycles=1,
        synergy_cycles=1,
    )

    cli.full_autonomous_run(args, synergy_history=hist, synergy_ma_history=ma_hist)

    roi_file = Path(args.sandbox_data_dir) / "roi_history.json"
    data = json.loads(roi_file.read_text())
    assert data.get("roi_history") == [0.1]
    assert data.get("synergy_history") == [{"synergy_roi": 0.2}]


def test_visual_agent_busy_via_client(tmp_path, monkeypatch):
    """Starting a run and immediately sending another should return 409."""
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        import importlib
        vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))

        class DummyLock:
            def acquire(self, timeout=0, poll_interval=0.05):
                class G:
                    def __enter__(self_inner):
                        return self_inner
                    def __exit__(self_inner, exc_type, exc, tb):
                        pass
                return G()

            def release(self):
                pass

            @property
            def is_locked(self):
                return False

        monkeypatch.setattr(vac_mod, "_global_lock", DummyLock())

        client1 = vac_mod.VisualAgentClient(urls=[url], poll_interval=0.05, token="tombalolosvisualagent123")
        client2 = vac_mod.VisualAgentClient(urls=[url], poll_interval=0.05, token="tombalolosvisualagent123")

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

        fut = client1.ask_async([{"content": "a"}])
        time.sleep(0.01)
        client2.ask([{"content": "b"}])
        fut.result(timeout=5)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_run_endpoint_busy_with_testclient(monkeypatch, tmp_path):
    """Concurrent /run requests should return 409 for the second until the first finishes."""
    from tests.test_visual_agent_server import _setup_va
    import types, sys

    sys.modules.setdefault("psutil", types.ModuleType("psutil")).pid_exists = lambda *_a, **_k: False

    va = _setup_va(monkeypatch, tmp_path, start_worker=True)

    def fake_run(prompt: str, branch: str | None = None) -> None:
        time.sleep(0.2)

    monkeypatch.setattr(va, "run_menace_pipeline", fake_run)

    shared = threading.Lock()

    class DummyLock:
        def acquire(self, timeout: float = 0):
            if not shared.acquire(blocking=False):
                raise va.Timeout()

        def release(self):
            if shared.locked():
                shared.release()

        @property
        def is_locked(self):
            return shared.locked()

    monkeypatch.setattr(va, "_global_lock", DummyLock())

    with TestClient(va.app) as client:
        responses: dict[str, requests.Response] = {}

        def first():
            responses["r1"] = client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "a"})

        def second():
            time.sleep(0.01)
            responses["r2"] = client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "b"})

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert responses["r1"].status_code == 202
        assert responses["r2"].status_code == 202

        # Wait for the first job to complete
        for _ in range(20):
            status = client.get("/status").json()
            if not status.get("active") and status.get("queue") == 0:
                break
            time.sleep(0.05)
        resp3 = client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "c"})
        assert resp3.status_code == 202

    va._exit_event.set()
    va._worker_thread.join(timeout=1)
    va._autosave_thread.join(timeout=1)

    for p in tmp_path.glob("visual_agent_*"):
        p.unlink(missing_ok=True)


def test_task_endpoint_busy(tmp_path):
    """Concurrent /task requests should yield HTTP 409 for the second."""

    script = tmp_path / "server.py"  # path-ignore
    script.write_text(
        textwrap.dedent(
            """
            import sys, types, threading, importlib, time, os
            heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
            for name in heavy:
                sys.modules[name] = types.ModuleType(name)
            filelock_mod = types.ModuleType('filelock')
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
            sys.modules['filelock'] = filelock_mod
            pt_mod = types.ModuleType('pytesseract')
            pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
            pt_mod.image_to_string = lambda *a, **k: ''
            pt_mod.image_to_data = lambda *a, **k: {}
            pt_mod.Output = types.SimpleNamespace(DICT=0)
            sys.modules['pytesseract'] = pt_mod
            va = importlib.import_module('menace_visual_agent_2')
            def fake_run(prompt: str, branch: str | None = None):
                time.sleep(0.2)
            va.run_menace_pipeline = fake_run
            shared = threading.Lock()
            class DummyLock:
                def acquire(self, timeout: float = 0):
                    if not shared.acquire(blocking=False):
                        raise va.Timeout()
                def release(self):
                    if shared.locked():
                        shared.release()
                @property
                def is_locked(self):
                    return shared.locked()
            va._global_lock = DummyLock()
            va.app.add_api_route('/task', va.run_task, methods=['POST'], status_code=202)
            import uvicorn
            uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
            """
        )
    )

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_PID_FILE"] = str(tmp_path / "agent.pid")
    env["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "agent.lock")
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        responses: dict[str, requests.Response] = {}

        def first():
            responses['r1'] = requests.post(
                f"http://127.0.0.1:{port}/task",
                headers={"x-token": TOKEN},
                json={"prompt": "p"},
                timeout=5,
            )

        def second():
            time.sleep(0.01)
            responses['r2'] = requests.post(
                f"http://127.0.0.1:{port}/task",
                headers={"x-token": TOKEN},
                json={"prompt": "p"},
                timeout=5,
            )

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert responses['r1'].status_code == 202
        assert responses['r2'].status_code == 202
    finally:
        proc.terminate()
        proc.wait(timeout=5)

