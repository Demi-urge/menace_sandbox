import importlib
import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from tests.test_autonomous_integration import _free_port


def setup_full_stubs(monkeypatch, tmp_path: Path, captured: dict) -> None:
    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")
        class DummyLock:
            def __init__(self, *a, **k):
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

    def _ema(seq):
        if not seq:
            return 0.0, 0.0
        mean = sum(seq) / len(seq)
        var = sum((v - mean) ** 2 for v in seq) / len(seq)
        return mean, var ** 0.5

    def _adaptive_synergy_threshold(hist, window, *, factor=2.0, **_k):
        vals = [h.get("synergy_roi", 0.0) for h in hist[-window:]]
        if not vals:
            return 0.0
        _, std = _ema(vals)
        return std * factor

    def adaptive_synergy_convergence(hist, window, *, threshold=None, threshold_window=None, **_k):
        if threshold is None:
            threshold = _adaptive_synergy_threshold(hist, threshold_window or window)
        vals = [h.get("synergy_roi", 0.0) for h in hist[-window:]]
        if not vals:
            return False, 0.0, 0.0
        ema, _ = _ema(vals)
        return abs(ema) <= threshold, abs(ema), 0.99 if abs(ema) <= threshold else 0.5

    cli_stub._ema = _ema
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = _adaptive_synergy_threshold
    cli_stub.adaptive_synergy_convergence = adaptive_synergy_convergence
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})

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
        data["metrics_history"]["synergy_roi"].append(0.02 / (run_num ** 2))
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
    sw_mod = importlib.import_module("menace.synergy_weight_cli")
    def fake_train(history, path):
        captured["trained"] = list(history)
        Path(path).write_text(json.dumps({"count": len(history)}))
        return {}
    monkeypatch.setattr(sw_mod, "train_from_history", fake_train)
    class FastTrainer(sat_mod.SynergyAutoTrainer):
        def __init__(self, *a, **k):
            k.setdefault("interval", 0.05)
            super().__init__(*a, **k)
        def start(self):
            captured["trainer_started"] = True
            super().start()
        def stop(self):
            captured["trainer_stopped"] = True
            super().stop()
    monkeypatch.setattr(sat_mod, "SynergyAutoTrainer", FastTrainer)

    se_mod = importlib.import_module("menace.synergy_exporter")
    class TestExporter(se_mod.SynergyExporter):
        def __init__(self, *a, **k):
            k.setdefault("interval", 0.05)
            super().__init__(*a, **k)
            captured["exporter"] = self
    monkeypatch.setattr(se_mod, "SynergyExporter", TestExporter)
    captured["exporter_cls"] = TestExporter

    sym_mon = importlib.import_module("synergy_monitor")
    class DummyMon:
        def __init__(self, obj, *a, **k):
            self.obj = obj
            self.restart_count = 0
        def start(self):
            pass
        def stop(self):
            if hasattr(self.obj, "stop"):
                self.obj.stop()
    monkeypatch.setattr(sym_mon, "ExporterMonitor", DummyMon)
    monkeypatch.setattr(sym_mon, "AutoTrainerMonitor", DummyMon)
    monkeypatch.setitem(sys.modules, "ExporterMonitor", DummyMon)
    monkeypatch.setitem(sys.modules, "AutoTrainerMonitor", DummyMon)


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


def test_run_autonomous_integration_full(monkeypatch, tmp_path: Path):
    captured: dict = {}
    setup_full_stubs(monkeypatch, tmp_path, captured)
    mod = load_module(monkeypatch)
    db_mod = importlib.import_module("menace.synergy_history_db")
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "SynergyExporter", captured["exporter_cls"])
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))
    monkeypatch.setattr(mod, "shd", types.SimpleNamespace(connect_locked=lambda p: db_mod.connect(p)), raising=False)

    class DummySettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = 3
            self.synergy_cycles = 3
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
    monkeypatch.setenv("AUTO_TRAIN_INTERVAL", "0.05")
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.05")
    monkeypatch.setenv("SYNERGY_TRAINER_CHECK_INTERVAL", "0.05")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations", "1",
        "--runs", "3",
        "--preset-count", "1",
        "--sandbox-data-dir", str(tmp_path),
        "--no-recursive-orphans",
        "--no-recursive-isolated",
        "--include-orphans",
        "--discover-orphans",
        "--no-discover-isolated",
    ])

    conn = sqlite3.connect(tmp_path / "synergy_history.db")
    hist = db_mod.fetch_all(conn)
    conn.close()
    assert len(hist) == 3

    exp = captured.get("exporter")
    assert exp is not None
    gauge = exp._gauges.get("synergy_roi")
    if gauge is None:
        import time
        for _ in range(20):
            time.sleep(0.1)
            gauge = exp._gauges.get("synergy_roi")
            if gauge is not None:
                break
    if gauge is None:
        values = exp._load_latest()
        for name, val in values.items():
            g = exp._gauges.get(name)
            if g is None:
                from metrics_exporter import Gauge

                g = Gauge(name, f"Latest value for {name}")
                exp._gauges[name] = g
            g.set(float(val))
        gauge = exp._gauges.get("synergy_roi")
    assert gauge is not None
    assert gauge.labels().get() == pytest.approx(hist[-1]["synergy_roi"])

    from metrics_exporter import synergy_threshold_gauge

    thr_val = mod.sandbox_runner.cli._adaptive_synergy_threshold(hist, 3)
    assert synergy_threshold_gauge.labels().get() == pytest.approx(thr_val)

    conv, ema_val, conf = mod.sandbox_runner.cli.adaptive_synergy_convergence(hist, 3, threshold=thr_val)
    assert conv is True
    assert conf >= 0.5

    assert captured.get("trainer_started")
    assert captured.get("trainer_stopped")
    assert "trained" in captured and len(captured["trained"]) >= 1
