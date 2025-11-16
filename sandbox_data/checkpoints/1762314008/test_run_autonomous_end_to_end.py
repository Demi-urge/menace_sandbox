import importlib
import json
import sqlite3
import types
from pathlib import Path
import pytest

from tests.test_autonomous_integration import (
    setup_stubs,
    load_module,
    _free_port,
)


def test_run_autonomous_end_to_end(monkeypatch, tmp_path: Path):
    captured: dict = {}
    setup_stubs(monkeypatch, tmp_path, captured)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "SynergyExporter", captured["exporter_cls"])
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))
    db_mod = importlib.import_module("menace.synergy_history_db")
    sym_mon = importlib.import_module("synergy_monitor")
    class DummyMon:
        def __init__(self, trainer, *a, **k) -> None:
            self.trainer = trainer
            self.restart_count = 0

        def start(self) -> None:
            pass

        def stop(self) -> None:
            if hasattr(self.trainer, "stop"):
                self.trainer.stop()

    monkeypatch.setattr(sym_mon, "ExporterMonitor", DummyMon)
    monkeypatch.setattr(sym_mon, "AutoTrainerMonitor", DummyMon)
    monkeypatch.setattr(mod, "ExporterMonitor", DummyMon)
    monkeypatch.setattr(mod, "AutoTrainerMonitor", DummyMon)
    monkeypatch.setattr(
        mod,
        "shd",
        types.SimpleNamespace(connect_locked=lambda p: db_mod.connect(p)),
        raising=False,
    )

    class DummySettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
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
    monkeypatch.setenv("AUTO_TRAIN_INTERVAL", "0.01")
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.01")
    monkeypatch.setenv("SYNERGY_TRAINER_CHECK_INTERVAL", "0.01")
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

    import time

    exp = captured.get("exporter")
    assert exp is not None
    assert (tmp_path / "synergy_history.db").exists()

    from metrics_exporter import roi_threshold_gauge, synergy_threshold_gauge

    assert roi_threshold_gauge.labels().get() is not None
    assert synergy_threshold_gauge.labels().get() is not None

    meta_log = tmp_path / "sandbox_meta.log"
    entries = [json.loads(l.split(" ", 1)[1]) for l in meta_log.read_text().splitlines()]
    assert any(e.get("event") == "auto_trainer_stopped" for e in entries)
    assert captured.get("trainer_started")
    assert captured.get("trainer_stopped")
