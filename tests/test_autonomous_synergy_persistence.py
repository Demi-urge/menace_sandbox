import importlib
import importlib.util
import json
import sqlite3
import socket
import time
import types
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


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


def _setup_stubs(monkeypatch, val: float):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1"}]
    tracker_mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": [val]}
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

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        data_dir = Path(args.sandbox_data_dir or "sandbox_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        with open(roi_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "roi_history": [0.1],
                    "module_deltas": {},
                    "metrics_history": {"synergy_roi": [val]},
                },
                fh,
            )

    cli_stub.full_autonomous_run = fake_run
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda *a, **k: DummyTracker()
    sr_stub.cli = cli_stub

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)
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
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
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


def _load_module(monkeypatch, tmp_path: Path):
    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", importlib.import_module)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
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

    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))
    return mod


def _run_once(monkeypatch, tmp_path: Path, val: float) -> None:
    _setup_stubs(monkeypatch, val)
    mod = _load_module(monkeypatch, tmp_path)
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
    me = importlib.import_module("menace.metrics_exporter")
    me.stop_metrics_server()


def test_synergy_persistence(monkeypatch, tmp_path: Path) -> None:
    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    _run_once(monkeypatch, tmp_path, 0.05)
    _run_once(monkeypatch, tmp_path, 0.1)

    hist_file = tmp_path / "synergy_history.db"
    conn = sqlite3.connect(hist_file)
    rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
    conn.close()
    data = [json.loads(r[0]) for r in rows]
    assert len(data) == 2
    assert data[0].get("synergy_roi") == 0.05
    assert data[1].get("synergy_roi") == 0.1

    se = importlib.import_module("menace.synergy_exporter")
    me = importlib.import_module("menace.metrics_exporter")
    port2 = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port2)
    exp.start()
    try:
        metrics = {}
        for _ in range(50):
            try:
                resp = urllib.request.urlopen(f"http://localhost:{port2}/metrics")
                metrics = _parse_metrics(resp.read().decode())
                if metrics.get("synergy_roi") == 0.1:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.1
    finally:
        exp.stop()
        me.stop_metrics_server()
