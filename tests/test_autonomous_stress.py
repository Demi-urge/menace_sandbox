import json
import os
import types
from pathlib import Path

import pytest

from tests.test_run_autonomous_env_vars import _load_module


@pytest.mark.stress
@pytest.mark.skipif(not os.getenv("RUN_STRESS_TESTS"), reason="stress testing disabled")
def test_autonomous_long_loop(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)

    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.visual_agent_autostart = False
            self.visual_agent_urls = ""
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

    # stub metrics exporter to avoid server startup
    metrics_stub = types.ModuleType("metrics_exporter")
    metrics_stub.start_metrics_server = lambda *a, **k: None
    metrics_stub.roi_threshold_gauge = types.SimpleNamespace(set=lambda v: None)
    metrics_stub.synergy_threshold_gauge = types.SimpleNamespace(set=lambda v: None)
    monkeypatch.setitem(mod.sys.modules, "metrics_exporter", metrics_stub)
    monkeypatch.setitem(mod.sys.modules, "sandbox_runner.metrics_exporter", metrics_stub)

    hist_path = tmp_path / "synergy_history.db"

    class _Conn:
        def __init__(self, path: Path):
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text("[]")

        def execute(self, _q):
            data = json.loads(self.path.read_text())
            return types.SimpleNamespace(fetchall=lambda: [(json.dumps(d),) for d in data])

        def close(self):
            pass

    def connect_locked(path):
        return _Conn(Path(path))

    def insert_entry(conn, entry):
        data = json.loads(conn.path.read_text())
        data.append({str(k): float(v) for k, v in entry.items()})
        conn.path.write_text(json.dumps(data))

    def load_previous_synergy(data_dir):
        path = Path(data_dir) / "synergy_history.db"
        if not path.exists():
            return [], []
        data = json.loads(path.read_text())
        hist = [{str(k): float(v) for k, v in d.items()} for d in data if isinstance(d, dict)]
        return hist, []

    mod.shd = types.SimpleNamespace(connect_locked=connect_locked)
    monkeypatch.setattr(mod, "connect_locked", connect_locked)
    monkeypatch.setattr(mod, "insert_entry", insert_entry)
    monkeypatch.setattr(mod, "load_previous_synergy", load_previous_synergy)
    monkeypatch.setattr(mod, "migrate_json_to_db", lambda *a, **k: None)

    call_idx = 0

    class DummyTracker:
        def __init__(self, roi_hist, metrics):
            self.roi_history = list(roi_hist)
            self.metrics_history = {k: list(v) for k, v in metrics.items()}
            self.module_deltas = {}

        def save_history(self, path):
            Path(path).write_text(json.dumps({
                "roi_history": self.roi_history,
                "module_deltas": self.module_deltas,
                "metrics_history": self.metrics_history,
            }))

        def load_history(self, path):
            pass

        def diminishing(self):
            return 0.0

    def fake_main(preset, args):
        nonlocal call_idx
        call_idx += 1
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        if roi_file.exists():
            data = json.loads(roi_file.read_text())
        else:
            data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
        data["roi_history"].append(0.1 * call_idx)
        data["metrics_history"]["synergy_roi"].append(0.05 * call_idx)
        roi_file.write_text(json.dumps(data))
        return DummyTracker(data["roi_history"], data["metrics_history"])

    sr = mod.sandbox_runner
    sr._sandbox_main = fake_main
    cli = sr.cli
    cli._diminishing_modules = lambda *a, **k: (set(), None)
    cli._ema = lambda seq: (0.0, [])
    cli._adaptive_threshold = lambda *a, **k: 0.0
    cli._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli._synergy_converged = lambda *a, **k: (False, 0.0, {})
    cli.adaptive_synergy_convergence = lambda *a, **k: (False, 0.0, {})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tok")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations",
        "50",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    roi_data = json.loads((tmp_path / "roi_history.json").read_text())
    syn_data = json.loads(hist_path.read_text())
    assert len(roi_data.get("roi_history", [])) == 50
    assert len(syn_data) == 50
