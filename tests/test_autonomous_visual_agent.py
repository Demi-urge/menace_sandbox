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


def load_module(monkeypatch=None):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    if monkeypatch is not None:
        monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        if "dotenv" not in sys.modules:
            dmod = types.ModuleType("dotenv")
            dmod.dotenv_values = lambda *a, **k: {}
            monkeypatch.setitem(sys.modules, "dotenv", dmod)
        if "pydantic" not in sys.modules:
            pmod = types.ModuleType("pydantic")
            class BaseModel:
                def __init__(self, **kw):
                    pass
            class ValidationError(Exception):
                pass
            def validator(*a, **k):
                def wrap(fn):
                    return fn
                return wrap
            pmod.BaseModel = BaseModel
            pmod.ValidationError = ValidationError
            pmod.validator = validator
            monkeypatch.setitem(sys.modules, "pydantic", pmod)
        else:
            BaseModel = sys.modules["pydantic"].BaseModel
        if "pydantic_settings" not in sys.modules:
            ps_mod = types.ModuleType("pydantic_settings")
            ps_mod.BaseSettings = BaseModel
            monkeypatch.setitem(sys.modules, "pydantic_settings", ps_mod)
    spec.loader.exec_module(mod)
    return mod


def setup_stubs(monkeypatch):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{}]
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


@pytest.mark.xfail(reason="complex dependencies", strict=False)
def test_run_autonomous_with_visual_agent(monkeypatch, tmp_path):
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
        monkeypatch.setattr(mod, "validate_presets", lambda p: p)

        def fake_full_run(args, *, synergy_history=None, synergy_ma_history=None):
            headers = {"x-token": "tombalolosvisualagent123"}
            res1 = requests.post(f"{url}/run", headers=headers, json={"prompt": "a"})
            assert res1.status_code == 202
            res2 = requests.post(f"{url}/run", headers=headers, json={"prompt": "b"})
            assert res2.status_code == 202
            time.sleep(0.3)
            res3 = requests.post(f"{url}/run", headers=headers, json={"prompt": "b"})
            assert res3.status_code == 202
            time.sleep(0.3)
            results = [res1.json(), res3.json()]
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
            (data_dir / "results.json").write_text(json.dumps(results))

        monkeypatch.setattr(mod, "full_autonomous_run", fake_full_run)
        monkeypatch.setattr(mod.sandbox_runner.cli, "full_autonomous_run", fake_full_run)

        mod.main([
            "--max-iterations", "1",
            "--runs", "1",
            "--preset-count", "1",
            "--sandbox-data-dir", str(tmp_path),
        ])

        data = json.loads((tmp_path / "visual_agent_state.json").read_text())
        assert len(data["status"]) == 2
        assert all(v["status"] == "completed" for v in data["status"].values())
        assert (tmp_path / "results.json").exists()
    finally:
        proc.terminate()
        proc.wait(timeout=5)



def test_client_enqueue_on_failure(monkeypatch, tmp_path):
    pkg_path = ROOT
    pkg_spec = importlib.util.spec_from_file_location(
        "menace", pkg_path / "__init__.py", submodule_search_locations=[str(pkg_path)]  # path-ignore
    )
    menace_pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["menace"] = menace_pkg
    pkg_spec.loader.exec_module(menace_pkg)
    vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    def bad_post(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        vac_mod,
        "requests",
        types.SimpleNamespace(post=bad_post, get=lambda *a, **k: None),
    )

    client = vac_mod.VisualAgentClient(urls=["http://x"], poll_interval=0.01)
    client.ask([{"content": "hi"}])

    queue_path = tmp_path / "visual_agent_client_queue.jsonl"
    assert queue_path.exists()
    data = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert data and data[0]["action"] == "run"


def test_local_queue_flush(monkeypatch, tmp_path):
    pkg_path = ROOT
    pkg_spec = importlib.util.spec_from_file_location(
        "menace",
        pkg_path / "__init__.py",  # path-ignore
        submodule_search_locations=[str(pkg_path)],
    )
    menace_pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["menace"] = menace_pkg
    pkg_spec.loader.exec_module(menace_pkg)
    vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    calls = {"fail": True, "sent": 0}

    def post(url, headers=None, json=None, timeout=10):
        if calls["fail"]:
            raise RuntimeError("boom")
        calls["sent"] += 1
        return types.SimpleNamespace(status_code=202, json=lambda: {}, text="")

    def get(url, timeout=10):
        return types.SimpleNamespace(status_code=200, json=lambda: {"active": False, "status": "done"})

    monkeypatch.setattr(vac_mod, "requests", types.SimpleNamespace(post=post, get=get))
    monkeypatch.setattr(vac_mod, "log_event", lambda *a, **k: "id")
    monkeypatch.setattr(vac_mod.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(vac_mod.VisualAgentClient, "_poll", lambda self, base: (True, "ok"))

    client = vac_mod.VisualAgentClient(urls=["http://x"], poll_interval=0.01, flush_interval=0.05)

    client.ask([{"content": "hi"}])

    queue_path = tmp_path / "visual_agent_client_queue.jsonl"
    assert queue_path.exists() and queue_path.read_text().strip()

    calls["fail"] = False

    client.flush_local_queue()

    assert not queue_path.read_text().strip()
    assert calls["sent"] >= 1
