import os
import sys
import types
from pathlib import Path
import pytest

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

# provide minimal menace package for relative imports
ROOT = Path(__file__).resolve().parents[1]
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", menace_pkg)
log_stub = types.ModuleType("menace.logging_utils")
log_stub.get_logger = lambda *a, **k: types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
sys.modules.setdefault("menace.logging_utils", log_stub)

stub = types.ModuleType("stub")
stats_stub = types.SimpleNamespace(
    pearsonr=lambda *a, **k: (0.0, 0.0),
    t=lambda *a, **k: 0.0,
)
stub.stats = stats_stub
stub.isscalar = lambda x: isinstance(x, (int, float))
stub.bool_ = bool
sys.modules.setdefault("scipy", stub)
sys.modules.setdefault("scipy.stats", stats_stub)
sys.modules.setdefault("yaml", stub)
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
lm_stub = types.ModuleType("sklearn.linear_model")
lm_stub.LinearRegression = lambda *a, **k: types.SimpleNamespace(
    fit=lambda *a, **k: None,
    predict=lambda X: [0.0],
)
pf_stub = types.ModuleType("sklearn.preprocessing")
pf_stub.PolynomialFeatures = lambda *a, **k: types.SimpleNamespace(
    fit_transform=lambda X: X
)
sys.modules.setdefault("sklearn.linear_model", lm_stub)
sys.modules.setdefault("sklearn.preprocessing", pf_stub)
import importlib.util
from pathlib import Path

class _StubPM:
    registry: dict = {}

    def get_prediction_bots_for(self, name: str) -> list:
        return []

    def assign_prediction_bots(self, tracker):
        return []

pm_mod = types.ModuleType("menace.prediction_manager_bot")
pm_mod.PredictionManager = _StubPM
sys.modules.setdefault("menace.prediction_manager_bot", pm_mod)

spec = importlib.util.spec_from_file_location("menace.roi_tracker", Path("roi_tracker.py"))  # path-ignore
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
sys.modules.setdefault("menace.roi_tracker", rt)


# Provide a minimal Flask stub to satisfy metrics_dashboard import
flask_stub = types.ModuleType("flask")
class _DummyFlask:
    def add_url_rule(self, *a, **k):
        pass
    def run(self, host="0.0.0.0", port=0):
        pass
flask_stub.Flask = lambda *a, **k: _DummyFlask()
flask_stub.jsonify = lambda *a, **k: {}
sys.modules.setdefault("flask", flask_stub)

# Minimal stub for MetricsDashboard import
dash_stub = types.ModuleType("menace.metrics_dashboard")

class _DummyDash:
    def __init__(self, *a, **k):
        pass

    def run(self, host="0.0.0.0", port=0):
        pass

dash_stub.MetricsDashboard = _DummyDash
sys.modules.setdefault("menace.metrics_dashboard", dash_stub)

import argparse
import sandbox_runner.cli as cli


class DummyTracker:
    def __init__(self):
        self.module_deltas = {}
        self.metrics_history = {}
        self.predicted_roi = []
        self.actual_roi = []
        self.predicted_metrics = {}
        self.actual_metrics = {}

    def update(self, delta, synergy):
        self.module_deltas.setdefault("mod", []).append(delta)
        self.metrics_history.setdefault("synergy_roi", []).append(synergy)

    def diminishing(self):
        return 0.01

    def rankings(self):
        total = sum(self.module_deltas.get("mod", []))
        return [("mod", total, total)]


def test_autonomous_synergy_loop(monkeypatch):
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.05, 0.05),
        (0.05, 0.05),
        (0.005, 0.005),
        (0.005, 0.005),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=10,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert len(calls) == 4
    assert tracker.module_deltas["mod"] == [0.05, 0.05, 0.005, 0.005]
    assert tracker.metrics_history["synergy_roi"] == [0.05, 0.05, 0.005, 0.005]


def test_autonomous_synergy_converges(monkeypatch):
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.1, 0.1),
        (0.1, 0.1),
        (0.02, 0.02),
        (0.02, 0.02),
        (0.005, 0.005),
        (0.005, 0.005),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    log_calls = []

    def fake_info(msg, *a, **k):
        log_calls.append((msg, k.get("extra")))

    monkeypatch.setattr(cli.logger, "info", fake_info)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=10,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert len(calls) == 6
    assert tracker.module_deltas["mod"] == [0.1, 0.1, 0.02, 0.02, 0.005, 0.005]
    assert tracker.metrics_history["synergy_roi"] == [0.1, 0.1, 0.02, 0.02, 0.005, 0.005]
    msg = ("synergy convergence reached", {"iteration": 3, "ema": pytest.approx(0.005)})
    assert msg in log_calls


def test_async_synergy_prediction(monkeypatch):
    """Ensure async synergy prediction stores pred_synergy values."""

    from plugins import synergy_predict as sp
    from menace.roi_tracker import ROITracker
    pm = _StubPM()
    import asyncio

    tracker = ROITracker()
    sp.register(pm, tracker)

    tracker.update(0.0, 0.1, metrics={"synergy_roi": 0.05})
    metrics = asyncio.run(sp.collect_metrics_async(0.0, 0.1, None))
    tracker.register_metrics(*metrics.keys())
    tracker.update(0.1, 0.2, metrics=metrics)

    assert "pred_synergy_roi" in tracker.metrics_history


def test_autonomous_synergy_no_convergence(monkeypatch):
    """Oscillating synergy values should prevent convergence."""

    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.1, 0.1),
        (0.1, -0.1),
        (0.02, 0.1),
        (0.02, -0.1),
        (0.005, 0.1),
        (0.005, -0.1),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    log_calls = []

    def fake_info(msg, *a, **k):
        log_calls.append(msg)

    monkeypatch.setattr(cli.logger, "info", fake_info)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=3,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert "synergy convergence reached" not in log_calls


def test_autonomous_synergy_spike(monkeypatch):
    """Sudden synergy spike should reset convergence."""

    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.1, 0.05),
        (0.1, 0.04),
        (0.02, 0.03),
        (0.02, 0.02),
        (0.005, 0.009),
        (0.005, 1.0),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    log_calls = []

    def fake_info(msg, *a, **k):
        log_calls.append(msg)

    monkeypatch.setattr(cli.logger, "info", fake_info)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=3,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert "synergy convergence reached" not in log_calls

def test_run_autonomous_synergy_files(monkeypatch, tmp_path):
    """run_autonomous should update synergy history and weights files"""
    import importlib
    import json
    import sqlite3
    from tests.test_autonomous_integration import setup_stubs, load_module, _free_port

    captured = {}
    setup_stubs(monkeypatch, tmp_path, captured)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))
    db_mod = importlib.import_module("menace.synergy_history_db")
    def _direct_conn(path):
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
        )
        return conn
    monkeypatch.setattr(db_mod, "connect_locked", lambda p: _direct_conn(p), raising=False)
    monkeypatch.setattr(mod, "shd", db_mod, raising=False)

    # lightweight exporter stub
    se_mod = importlib.import_module("menace.synergy_exporter")
    class DummyExporter:
        def __init__(self, *a, **k):
            pass
        def start(self):
            pass
        def stop(self):
            pass
    monkeypatch.setattr(se_mod, "SynergyExporter", DummyExporter)

    # trainer stub storing instance for later
    sat_mod = importlib.import_module("menace.synergy_auto_trainer")
    class DummyTrainer:
        def __init__(self, *, history_file, weights_file, **_k):
            self.history_file = Path(history_file)
            self.weights_file = Path(weights_file)
            captured["trainer"] = self
        def start(self):
            pass
        def stop(self):
            pass
        def run_once(self):
            if self.history_file.exists():
                conn = sqlite3.connect(self.history_file)
                rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
                conn.close()
                data = [json.loads(r[0]) for r in rows]
            else:
                data = []
            self.weights_file.write_text(json.dumps({"count": len(data)}))
    monkeypatch.setattr(sat_mod, "SynergyAutoTrainer", DummyTrainer)

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
    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("AUTO_TRAIN_SYNERGY", "1")
    monkeypatch.setenv("AUTO_TRAIN_INTERVAL", "0.05")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    mod.main([
        "--max-iterations", "1",
        "--runs", "1",
        "--preset-count", "1",
        "--sandbox-data-dir", str(tmp_path),
    ])

    conn = sqlite3.connect(tmp_path / "synergy_history.db")
    rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
    conn.close()
    assert rows
    entry = json.loads(rows[-1][0])
    assert entry.get("synergy_roi") == 0.05

    trainer = captured.get("trainer")
    assert trainer is not None
    trainer.run_once()

    weights_path = tmp_path / "synergy_weights.json"
    assert weights_path.exists()
    weights = json.loads(weights_path.read_text())
    assert weights.get("count") == len(rows)
