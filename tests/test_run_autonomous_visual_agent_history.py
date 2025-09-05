import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import types
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")

TOKEN = "tombalolosvisualagent123"
ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")
        class DummyLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
        fl.FileLock = DummyLock
        fl.Timeout = RuntimeError
        monkeypatch.setitem(sys.modules, "filelock", fl)
    if "dotenv" not in sys.modules:
        dmod = types.ModuleType("dotenv")
        dmod.dotenv_values = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, "dotenv", dmod)
    spec.loader.exec_module(mod)
    return mod


def setup_stubs(monkeypatch):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]
    tracker_mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": [0.05]}
            self.roi_history = [0.1]

        def save_history(self, path):
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                    }
                )
            )

        def load_history(self, path):
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args, **k: None
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = (
        lambda hist, win, *, threshold=None, threshold_window=None, **k: (
            True,
            0.0,
            {},
        )
    )
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))


def _start_server(tmp_path: Path):
    script = tmp_path / "server.py"  # path-ignore
    script.write_text(
        textwrap.dedent(
            """
            import sys, types, importlib, time, os, threading
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
            sys.modules['psutil'] = types.SimpleNamespace(pid_exists=lambda *_a, **_k: False)
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
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    pass
            filelock_mod.FileLock = DummyFileLock
            filelock_mod.Timeout = DummyTimeout
            sys.modules['filelock'] = filelock_mod
            os.environ['VISUAL_AGENT_LOCK_FILE'] = os.path.join(
                os.environ.get('SANDBOX_DATA_DIR', '.'), 'server.lock')
            import menace_visual_agent_2 as va
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
            import uvicorn
            uvicorn.run(
                va.app,
                host='127.0.0.1',
                port=int(os.environ['MENACE_AGENT_PORT']),
                log_level='error')
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
    env["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "server.lock")
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


def test_run_autonomous_histories_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(tmp_path / "server.lock"))
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        setup_stubs(monkeypatch)
        monkeypatch.setenv("VISUAL_AGENT_URLS", url)
        monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
        monkeypatch.chdir(tmp_path)
        mod = load_module(monkeypatch)
        monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

        def fake_full_run(args, *, synergy_history=None, synergy_ma_history=None):
            headers = {"x-token": TOKEN}
            res1 = requests.post(f"{url}/run", headers=headers, json={"prompt": "a"})
            assert res1.status_code == 202
            _ = requests.post(f"{url}/run", headers=headers, json={"prompt": "b"})
            time.sleep(0.3)
            res3 = requests.post(f"{url}/run", headers=headers, json={"prompt": "b"})
            assert res3.status_code == 202
            time.sleep(0.3)
            data_dir = Path(args.sandbox_data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "roi_history.json").write_text(
                json.dumps(
                    {
                        "roi_history": [0.1],
                        "module_deltas": {},
                        "metrics_history": {"synergy_roi": [0.05]},
                    }
                )
            )
            (data_dir / "synergy_history.json").write_text(
                json.dumps([{"synergy_roi": 0.05}])
            )

        monkeypatch.setattr(mod, "full_autonomous_run", fake_full_run)
        monkeypatch.setattr(mod.sandbox_runner.cli, "full_autonomous_run", fake_full_run)

        mod.main([
            "--max-iterations", "1",
            "--runs", "1",
            "--preset-count", "1",
            "--sandbox-data-dir", str(tmp_path),
        ])

        roi_file = tmp_path / "roi_history.json"
        synergy_file = tmp_path / "synergy_history.json"
        assert roi_file.exists() and synergy_file.exists()
        roi_data = json.loads(roi_file.read_text())
        syn_data = json.loads(synergy_file.read_text())
        assert roi_data.get("roi_history") == [0.1]
        assert syn_data == [{"synergy_roi": 0.05}]

        import sqlite3
        with sqlite3.connect(tmp_path / "visual_agent_queue.db") as conn:
            rows = conn.execute("SELECT status FROM tasks").fetchall()
        assert len(rows) == 2
        assert all(r[0] == "completed" for r in rows)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
