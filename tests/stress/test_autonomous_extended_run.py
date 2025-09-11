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
def test_autonomous_extended_run(monkeypatch, tmp_path: Path) -> None:
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

    sat_mod = importlib.import_module("menace.synergy_auto_trainer")

    db_mod = importlib.import_module("menace.synergy_history_db")
    monkeypatch.setattr(mod, "connect_locked", db_mod.connect)
    monkeypatch.setattr(mod, "insert_entry", db_mod.insert_entry)
    monkeypatch.setattr(mod, "migrate_json_to_db", db_mod.migrate_json_to_db)
    monkeypatch.setattr(db_mod, "connect_locked", db_mod.connect)
    monkeypatch.setattr(mod, "shd", db_mod, raising=False)

    sr_stub = importlib.import_module("sandbox_runner")
    cli_stub = importlib.import_module("sandbox_runner.cli")

    def looped_run(args, **_k):
        iters = int(getattr(args, "max_iterations", 1) or 1)
        for _ in range(iters):
            sr_stub._sandbox_main({}, args)

    monkeypatch.setattr(cli_stub, "full_autonomous_run", looped_run)
    monkeypatch.setattr(mod, "full_autonomous_run", looped_run)

    class CapturingTrainer:
        def __init__(self, *a, **k):
            captured["trainer_init"] = True

        def start(self) -> None:
            captured["trainer_started"] = True

        def stop(self) -> None:
            captured["trainer_stopped"] = True

    monkeypatch.setattr(sat_mod, "SynergyAutoTrainer", CapturingTrainer)

    sym_mon = importlib.import_module("synergy_monitor")

    monkeypatch.setattr(
        sym_mon,
        "ExporterMonitor",
        lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0),
    )
    monkeypatch.setattr(
        mod,
        "ExporterMonitor",
        lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0),
    )
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
    monkeypatch.setenv("AUTO_TRAIN_SYNERGY", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(exporter_port))
    monkeypatch.setenv("AUTO_TRAIN_INTERVAL", "0.05")
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.05")
    monkeypatch.setenv("SYNERGY_TRAINER_CHECK_INTERVAL", "0.05")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main(
        [
            "--max-iterations",
            "501",
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
    assert len(roi_data.get("roi_history", [])) >= 501

    hist_file = tmp_path / "synergy_history.db"
    db_mod = importlib.import_module("menace.synergy_history_db")
    conn = sqlite3.connect(hist_file)
    syn_data = db_mod.fetch_all(conn)
    conn.close()
    assert len(syn_data) >= 501

    assert captured.get("trainer_started")
