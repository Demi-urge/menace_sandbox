import importlib
import json
import os
import sqlite3
import types
import urllib.request
from pathlib import Path

import pytest

from tests.test_autonomous_integration import setup_stubs, load_module, _free_port


@pytest.mark.stress
@pytest.mark.skipif(
    not os.getenv("RUN_STRESS_TESTS"), reason="stress testing disabled"
)
def test_autonomous_long_run(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    setup_stubs(monkeypatch, tmp_path, captured)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    se_mod = importlib.import_module("menace.synergy_exporter")

    class CapturingExporter(se_mod.SynergyExporter):
        def __init__(self, *a, **k):
            k.setdefault("interval", 0.05)
            super().__init__(*a, **k)
            captured["exporter"] = self

    monkeypatch.setattr(mod, "SynergyExporter", CapturingExporter)
    monkeypatch.setattr(se_mod, "SynergyExporter", CapturingExporter)

    sym_mon = importlib.import_module("synergy_monitor")

    class CapturingMonitor(sym_mon.ExporterMonitor):
        def __init__(self, exporter, log, *a, **k):
            super().__init__(exporter, log, interval=0.05)

    monkeypatch.setattr(sym_mon, "ExporterMonitor", CapturingMonitor)
    monkeypatch.setattr(mod, "ExporterMonitor", CapturingMonitor)
    monkeypatch.setattr(
        sym_mon,
        "AutoTrainerMonitor",
        lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0),
    )
    monkeypatch.setattr(
        mod,
        "AutoTrainerMonitor",
        lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0),
    )

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

    metrics_port = _free_port()
    exporter_port = _free_port()
    monkeypatch.setenv("METRICS_PORT", str(metrics_port))
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(exporter_port))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.05")

    mod.main(
        [
            "--max-iterations",
            "200",
            "--runs",
            "1",
            "--preset-count",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    exporter = captured.get("exporter")
    assert exporter is not None and exporter.health_port is not None
    health_url = f"http://localhost:{exporter.health_port}/health"
    data = urllib.request.urlopen(health_url).read().decode()
    info = json.loads(data)
    assert info.get("healthy") is True

    roi_file = tmp_path / "roi_history.json"
    roi_data = json.loads(roi_file.read_text())
    assert len(roi_data.get("roi_history", [])) >= 200

    hist_file = tmp_path / "synergy_history.db"
    db_mod = importlib.import_module("menace.synergy_history_db")
    conn = sqlite3.connect(hist_file)
    syn_data = db_mod.fetch_all(conn)
    conn.close()
    assert len(syn_data) >= 200

