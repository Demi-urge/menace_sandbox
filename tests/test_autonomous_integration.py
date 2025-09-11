import importlib.util
import importlib
import json
import os
import socket
import sqlite3
import sys
import types
from pathlib import Path


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def setup_stubs(monkeypatch, tmp_path: Path, captured: dict):
    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")
        class DummyLock:
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
        class _Root(object):
            @classmethod
            def __class_getitem__(cls, item):
                return cls

        pyd.RootModel = _Root
        pyd.ValidationError = type("ValidationError", (Exception,), {})
        pyd.validator = lambda *a, **k: (lambda f: f)
        pyd.Field = lambda default=None, **k: default
        pyd.BaseSettings = object
        monkeypatch.setitem(sys.modules, "pydantic", pyd)
    if "pydantic_settings" not in sys.modules:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings = object
        ps.SettingsConfigDict = dict
        monkeypatch.setitem(sys.modules, "pydantic_settings", ps)

    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]

    tracker_mod = types.ModuleType("menace.roi_tracker")
    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": []}
            self.roi_history = []
        def load_history(self, path):
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
                self.roi_history = data.get("roi_history", [])
                self.metrics_history = data.get("metrics_history", {})
        def diminishing(self):
            return 0.0
    tracker_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args, **k: sr_stub._sandbox_main({}, args)
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})

    def fake_sandbox_main(preset, args):
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        hist_file = data_dir / "roi_history.json"
        if hist_file.exists():
            data = json.loads(hist_file.read_text())
        else:
            data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
        run_num = len(data["roi_history"]) + 1
        data["roi_history"].append(0.1 * run_num)
        data["metrics_history"]["synergy_roi"].append(0.05 * run_num)
        hist_file.write_text(json.dumps(data))
        tracker = DummyTracker()
        tracker.roi_history = data["roi_history"]
        tracker.metrics_history = data["metrics_history"]
        return tracker

    sr_stub._sandbox_main = fake_sandbox_main
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))

    import metrics_exporter
    class DummyGauge:
        def __init__(self, *a, **k):
            self.value = 0.0
        def labels(self, *a, **k):
            return self
        def set(self, v):
            self.value = float(v)
        def inc(self, a=1.0):
            self.value += a
        def dec(self, a=1.0):
            self.value -= a
        def get(self):
            return self.value
    monkeypatch.setattr(metrics_exporter, "Gauge", DummyGauge, raising=False)
    metrics_exporter.roi_forecast_gauge = DummyGauge()
    metrics_exporter.synergy_forecast_gauge = DummyGauge()
    metrics_exporter.roi_threshold_gauge = DummyGauge()
    metrics_exporter.synergy_threshold_gauge = DummyGauge()
    metrics_exporter.synergy_adaptation_actions_total = DummyGauge()

    sat_mod = importlib.import_module("menace.synergy_auto_trainer")
    class DummyTrainer:
        def __init__(self, *a, **k):
            captured["trainer_init"] = True
        def start(self):
            captured["trainer_started"] = True
        def stop(self):
            captured["trainer_stopped"] = True
    monkeypatch.setattr(sat_mod, "SynergyAutoTrainer", DummyTrainer)

    se_mod = importlib.import_module("menace.synergy_exporter")
    class TestExporter(se_mod.SynergyExporter):
        def __init__(self, *a, **k):
            k.setdefault("interval", 0.05)
            super().__init__(*a, **k)
            captured["exporter"] = self
    monkeypatch.setattr(se_mod, "SynergyExporter", TestExporter)
    captured["exporter_cls"] = TestExporter


def load_module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"  # path-ignore
    sys.modules.pop("run_autonomous", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    return mod


def test_autonomous_with_exporter(monkeypatch, tmp_path: Path):
    captured: dict = {}
    setup_stubs(monkeypatch, tmp_path, captured)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "SynergyExporter", captured["exporter_cls"])
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    class DummySettings:
        def __init__(self) -> None:
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
    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("AUTO_TRAIN_SYNERGY", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.05")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations", "1",
        "--runs", "1",
        "--preset-count", "1",
        "--sandbox-data-dir", str(tmp_path),
        "--include-orphans",
        "--discover-orphans",
        "--no-discover-isolated",
    ])

    db_mod = importlib.import_module("menace.synergy_history_db")
    conn = sqlite3.connect(tmp_path / "synergy_history.db")
    hist = db_mod.fetch_all(conn)
    conn.close()
    assert hist and hist[-1]["synergy_roi"] == 0.05

    exp = captured.get("exporter")
    assert exp is not None
    gauge = exp._gauges.get("synergy_roi")
    assert gauge is not None
    assert gauge.labels().get() == 0.05
