import importlib
import json
import sqlite3
import socket
import time
import urllib.request
from pathlib import Path
import os
import sys
import subprocess
import signal
import pytest
from dynamic_path_router import resolve_path


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split()
        metrics[name] = float(value)
    return metrics


def test_exporter_serves_latest_metrics(tmp_path: Path) -> None:
    se = importlib.import_module("menace.synergy_exporter")
    me = importlib.import_module("menace.metrics_exporter")

    history = [
        {"synergy_roi": 0.2, "synergy_efficiency": 0.4},
        {"synergy_roi": 0.3, "synergy_efficiency": 0.5},
    ]
    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    for entry in history:
        conn.execute(
            "INSERT INTO synergy_history(entry) VALUES (?)",
            (json.dumps(entry),),
        )
    conn.commit()
    conn.close()

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)
    exp.start()
    try:
        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(
                    f"http://localhost:{port}/metrics"
                ).read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.3:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.3
        assert metrics.get("synergy_efficiency") == 0.5

        status = urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health"
        ).getcode()
        assert status == 200
    finally:
        exp.stop()
        me.stop_metrics_server()

    assert exp._thread is not None
    assert not exp._thread.is_alive()


@pytest.mark.skip("requires full environment")
def test_run_autonomous_starts_exporter(monkeypatch, tmp_path: Path) -> None:
    import importlib.util
    import sys
    import types
    import shutil

    # prepare temporary history database with initial data
    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.05}),),
    )
    conn.commit()
    conn.close()

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1"}]
    tracker_mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k) -> None:
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": [0.05]}
            self.roi_history = [0.1]

        def save_history(self, path: str) -> None:
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                    }
                )
            )

        def load_history(self, path: str) -> None:
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self) -> float:
            return 0.0

    tracker_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda *a, **k: None
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub.cli = cli_stub
    sr_stub._sandbox_main = lambda *a, **k: DummyTracker()

    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
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
    if "pydantic" not in sys.modules:
        pyd = types.ModuleType("pydantic")
        pyd.BaseModel = object
        pyd.ValidationError = type("ValidationError", (Exception,), {})
        pyd.validator = lambda *a, **k: (lambda f: f)
        pyd.BaseSettings = object
        pyd.Field = lambda default=None, **k: default
        monkeypatch.setitem(sys.modules, "pydantic", pyd)
    if "pydantic_settings" not in sys.modules:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings = object
        ps.SettingsConfigDict = dict
        monkeypatch.setitem(sys.modules, "pydantic_settings", ps)

    se_mod = importlib.import_module("menace.synergy_exporter")
    captured: dict[str, se_mod.SynergyExporter] = {}

    class TestExporter(se_mod.SynergyExporter):
        def __init__(self, history_file: str | Path, *, port: int, interval: float = 0.05) -> None:
            super().__init__(history_file, interval=interval, port=port)
            captured["exp"] = self

    monkeypatch.setattr(se_mod, "SynergyExporter", TestExporter)

    path = resolve_path("run_autonomous.py")  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SynergyExporter", TestExporter)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "start_metrics_server", lambda *a, **k: None)
    shd_stub = types.SimpleNamespace(connect_locked=lambda path: sqlite3.connect(path))
    monkeypatch.setattr(mod, "shd", shd_stub)
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
            self.synergy_cycles = None
            self.roi_threshold = None
            self.synergy_threshold = None
            self.roi_confidence = None
            self.synergy_confidence = None
            self.synergy_threshold_window = None
            self.synergy_threshold_weight = None
            self.synergy_ma_window = None
            self.synergy_stationarity_confidence = None
            self.synergy_std_threshold = None
            self.synergy_variance_confidence = None
            self.enable_relevancy_radar = False

        def __getattr__(self, _name: str):
            return None
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    exp = captured["exp"]
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://localhost:{port}/metrics").read()
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://localhost:{exp.health_port}/health")
    assert exp._thread is not None
    assert not exp._thread.is_alive()

    me = importlib.import_module("menace.metrics_exporter")
    me.stop_metrics_server()


def test_cli_standalone(monkeypatch, tmp_path: Path) -> None:
    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO synergy_history(entry) VALUES (?)",
        (json.dumps({"synergy_roi": 0.42}),),
    )
    conn.commit()
    conn.close()

    port = _free_port()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    synergy_exporter_path = str(resolve_path("synergy_exporter.py"))  # path-ignore
    script = f"""
import importlib.util, sys, types
pkg = types.ModuleType('menace')
pkg.__path__ = [r'{root}']
sys.modules['menace'] = pkg
spec = importlib.util.spec_from_file_location('menace.synergy_exporter', r'{synergy_exporter_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.main(['--history-file', r'{hist_file}', '--port', '{port}', '--interval', '0.05'])
"""
    proc = subprocess.Popen([sys.executable, "-c", script], env=env)
    try:
        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.42:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.42
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    assert proc.returncode == 0
