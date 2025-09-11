import importlib
import sys
import json
import os
import signal
import socket
import sqlite3
import subprocess
import time
import threading
import urllib.request
from pathlib import Path
from tests.test_exporter_restart import setup_stubs


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


def _load_run_autonomous(monkeypatch, tmp_path: Path):
    import importlib.util
    import sys
    import shutil

    setup_stubs(monkeypatch, tmp_path)

    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"  # path-ignore
    sys.modules.pop("run_autonomous", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    return mod


def test_exporter_updates_after_training(monkeypatch, tmp_path: Path) -> None:
    sat = importlib.import_module("menace.synergy_auto_trainer")
    se = importlib.import_module("menace.synergy_exporter")
    db_mod = importlib.import_module("menace.synergy_history_db")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.1})
    conn.close()

    weights_file = tmp_path / "weights.json"
    weights_file.write_text("{}")

    calls: list[tuple[list[dict[str, float]], Path]] = []

    def fake_train(history, path):
        calls.append((list(history), Path(path)))
        return {}

    monkeypatch.setattr(sat.synergy_weight_cli, "train_from_history", fake_train)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file,
        weights_file=weights_file,
        interval=0.05,
        progress_file=tmp_path / "progress.json",
    )
    trainer.start()

    port = _free_port()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    script = f"""
import importlib.util, sys, os
root = r'{root}'
parent = os.path.dirname(root)
sys.path.insert(0, parent)
sys.path.insert(1, root)
spec = importlib.util.spec_from_file_location('menace', os.path.join(root, '__init__.py'))  # path-ignore
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.modules.setdefault('menace', mod)
spec = importlib.util.spec_from_file_location('menace.synergy_exporter', os.path.join(root, 'synergy_exporter.py'))  # path-ignore
mod2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod2)
mod2.main(['--history-file', r'{hist_file}', '--port', '{port}', '--interval', '0.05'])
"""
    proc = subprocess.Popen([sys.executable, "-c", script], env=env)
    try:
        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.1:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.1
        assert len(calls) >= 1

        conn = db_mod.connect(hist_file)
        db_mod.insert_entry(conn, {"synergy_roi": 0.2})
        conn.close()

        for _ in range(50):
            if len(calls) >= 2:
                break
            time.sleep(0.05)
        assert len(calls) >= 2

        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.2:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.2
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        trainer.stop()

    assert proc.returncode == 0
    assert trainer._thread is None

    assert calls
    first_hist, first_path = calls[0]
    assert first_path == weights_file
    assert first_hist[0]["synergy_roi"] == 0.1


def test_synergy_tools_command(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    db_mod = importlib.import_module("menace.synergy_history_db")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.3})
    conn.close()

    weights_file = tmp_path / "synergy_weights.json"
    weights_file.write_text("{}")

    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["EXPORT_SYNERGY_METRICS"] = "1"
    env["AUTO_TRAIN_SYNERGY"] = "1"
    env["AUTO_TRAIN_INTERVAL"] = "0.05"
    env["SYNERGY_METRICS_PORT"] = str(port)

    script = f"""
import importlib.util, sys, os, json
root = r'{root}'
parent = os.path.dirname(root)
sys.path.insert(0, parent)
sys.path.insert(1, root)
spec = importlib.util.spec_from_file_location('menace', os.path.join(root, '__init__.py'))  # path-ignore
menace_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(menace_mod)
sys.modules.setdefault('menace', menace_mod)
spec = importlib.util.spec_from_file_location('menace.synergy_auto_trainer', os.path.join(root, 'synergy_auto_trainer.py'))  # path-ignore
sat = importlib.util.module_from_spec(spec); spec.loader.exec_module(sat)
spec2 = importlib.util.spec_from_file_location('menace.synergy_exporter', os.path.join(root, 'synergy_exporter.py'))  # path-ignore
se = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(se)
import types
a_mod = types.ModuleType('menace.audit_trail')
class DummyTrail:
    def __init__(self, path, private_key=None):
        self.path = path
    def record(self, message):
        pass
a_mod.AuditTrail = DummyTrail
sys.modules['menace.audit_trail'] = a_mod
spec3 = importlib.util.spec_from_file_location('synergy_tools', os.path.join(root, 'synergy_tools.py'))  # path-ignore
tools = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(tools)
log_file = os.path.join(r'{tmp_path}', 'called.json')
def fake_train(history, path):
    with open(log_file, 'w') as fh:
        json.dump(dict(history=list(history), path=str(path)), fh)
    return dict()
sat.synergy_weight_cli.train_from_history = fake_train
tools.main(['--sandbox-data-dir', r'{tmp_path}'])
"""

    proc = subprocess.Popen([sys.executable, "-c", script], env=env, cwd=tmp_path)
    try:
        metrics = {}
        for _ in range(60):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.3:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.3
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)

    assert proc.returncode == 0
    call_path = tmp_path / "called.json"
    assert call_path.exists()
    data = json.loads(call_path.read_text())
    assert Path(data["path"]) == tmp_path / "synergy_weights.json"
    assert data["history"][0]["synergy_roi"] == 0.3


def test_exporter_and_trainer_restart(monkeypatch, tmp_path: Path) -> None:
    ra = _load_run_autonomous(monkeypatch, tmp_path)
    se_mod = importlib.import_module("menace.synergy_exporter")
    sat_mod = importlib.import_module("menace.synergy_auto_trainer")

    captured = {"exp": 0, "trainer": 0}

    class CrashExporter(se_mod.SynergyExporter):
        def start(self) -> None:  # type: ignore[override]
            captured["exp"] += 1
            self._thread = threading.Thread(target=lambda: None, daemon=True)
            self._thread.start()
            self._thread.join(0.01)

    class CrashTrainer(sat_mod.SynergyAutoTrainer):
        def start(self) -> None:  # type: ignore[override]
            captured["trainer"] += 1
            self._thread = threading.Thread(target=lambda: None, daemon=True)
            self._thread.start()
            self._thread.join(0.01)

    monkeypatch.setattr(se_mod, "SynergyExporter", CrashExporter)
    monkeypatch.setattr(sat_mod, "SynergyAutoTrainer", CrashTrainer)
    monkeypatch.setattr(ra, "SynergyExporter", CrashExporter)
    monkeypatch.setattr(ra, "SynergyAutoTrainer", CrashTrainer)
    monkeypatch.setattr(ra, "_check_dependencies", lambda *a, **_k: True)

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

    monkeypatch.setattr(ra, "SandboxSettings", DummySettings)
    monkeypatch.setattr(ra, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("AUTO_TRAIN_SYNERGY", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))
    monkeypatch.setenv("AUTO_TRAIN_INTERVAL", "0.01")
    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.01")
    monkeypatch.setenv("SYNERGY_TRAINER_CHECK_INTERVAL", "0.01")

    ra.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "2",
            "--preset-count",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    assert captured["exp"] >= 2
    assert captured["trainer"] >= 2

    meta_log = tmp_path / "sandbox_meta.log"
    entries = [
        json.loads(l.split(" ", 1)[1]) for l in meta_log.read_text().splitlines()
    ]
    exp_restarts = [
        e.get("restart_count") for e in entries if e.get("event") == "exporter_restarted"
    ]
    trainer_restarts = [
        e.get("restart_count") for e in entries if e.get("event") == "auto_trainer_restarted"
    ]
    assert exp_restarts and exp_restarts[-1] == len(exp_restarts)
    assert trainer_restarts and trainer_restarts[-1] == len(trainer_restarts)


def test_exporter_start_stop_restart(tmp_path: Path) -> None:
    se = importlib.import_module("menace.synergy_exporter")
    db_mod = importlib.import_module("menace.synergy_history_db")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    conn.close()

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)

    exp.start()
    exp.stop()
    assert exp._thread is None

    exp._stop.clear()
    exp.start()
    exp.stop()

    assert exp._thread is None
