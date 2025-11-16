import importlib
import json
import os
import types
from pathlib import Path

import pytest

from tests.test_autonomous_integration import setup_stubs, load_module, _free_port


@pytest.mark.stress
@pytest.mark.skipif(
    not os.getenv("RUN_STRESS_TESTS"), reason="stress testing disabled"
)
def test_run_autonomous_convergence(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    setup_stubs(monkeypatch, tmp_path, captured)

    sr_stub = importlib.import_module("sandbox_runner")
    cli_stub = importlib.import_module("sandbox_runner.cli")
    tracker_mod = importlib.import_module("menace.roi_tracker")

    def fake_sandbox_main(preset, args):
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        hist_file = data_dir / "roi_history.json"
        if hist_file.exists():
            data = json.loads(hist_file.read_text())
        else:
            data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
        run_num = len(data["roi_history"])
        roi_val = 0.1 * (0.8 ** run_num)
        syn_val = 0.05 * (0.5 ** run_num)
        data["roi_history"].append(roi_val)
        data["metrics_history"]["synergy_roi"].append(syn_val)
        hist_file.write_text(json.dumps(data))
        tracker = tracker_mod.ROITracker()
        tracker.roi_history = data["roi_history"]
        tracker.metrics_history = data["metrics_history"]
        return tracker

    def _ema(values):
        if not values:
            return 0.0, 0.0
        alpha = 2.0 / (len(values) + 1)
        ema = values[0]
        ema_sq = values[0] ** 2
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
            ema_sq = alpha * (v ** 2) + (1 - alpha) * ema_sq
        var = ema_sq - ema ** 2
        if var < 1e-12:
            var = 0.0
        return ema, var ** 0.5

    def _adaptive_threshold(values, window, factor=2.0):
        if not values:
            return 0.0
        vals = values[-window:]
        _, std = _ema(vals)
        return float(std * factor)

    def _adaptive_synergy_threshold(hist, window, *, factor=2.0, **_k):
        vals = [h.get("synergy_roi", 0.0) for h in hist[-window:]]
        if not vals:
            return 0.0
        _, std = _ema(vals)
        return float(std * factor)

    def adaptive_synergy_convergence(hist, window, *, threshold=None, threshold_window=None, **_k):
        if threshold is None:
            threshold = _adaptive_synergy_threshold(hist, threshold_window or window)
        vals = [h.get("synergy_roi", 0.0) for h in hist[-window:]]
        if not vals:
            return False, 0.0, 0.0
        ema, _ = _ema(vals)
        if len(hist) < 50:
            return False, abs(ema), 0.5
        return True, abs(ema), 1.0

    monkeypatch.setattr(sr_stub, "_sandbox_main", fake_sandbox_main)
    monkeypatch.setattr(cli_stub, "_capture_run", lambda args, **k: sr_stub._sandbox_main({}, args))
    monkeypatch.setattr(cli_stub, "_adaptive_threshold", _adaptive_threshold)
    monkeypatch.setattr(cli_stub, "_adaptive_synergy_threshold", _adaptive_synergy_threshold)
    monkeypatch.setattr(cli_stub, "adaptive_synergy_convergence", adaptive_synergy_convergence)

    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    class DummySettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = False
            self.roi_cycles = 5
            self.synergy_cycles = 5
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
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.05")

    mod.main([
        "--max-iterations",
        "60",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    log_file = tmp_path / "threshold_log.jsonl"
    assert log_file.exists()
    entries = [json.loads(l) for l in log_file.read_text().splitlines()]
    assert len(entries) >= 50
    roi_vals = [e.get("roi_threshold") for e in entries if e.get("roi_threshold") is not None]
    assert roi_vals[0] > roi_vals[-1]
    assert any(e.get("converged") for e in entries)
