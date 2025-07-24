import importlib
import sys
import json
import os
import signal
import socket
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path


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

    called = {"count": 0}

    def fake_cli(args: list[str]) -> int:
        called["count"] += 1
        return 0

    monkeypatch.setattr(sat.synergy_weight_cli, "cli", fake_cli)

    trainer = sat.SynergyAutoTrainer(
        history_file=hist_file, weights_file=weights_file, interval=0.05
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
spec = importlib.util.spec_from_file_location('menace', os.path.join(root, '__init__.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.modules.setdefault('menace', mod)
spec = importlib.util.spec_from_file_location('menace.synergy_exporter', os.path.join(root, 'synergy_exporter.py'))
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
        assert called["count"] >= 1

        conn = db_mod.connect(hist_file)
        db_mod.insert_entry(conn, {"synergy_roi": 0.2})
        conn.close()

        for _ in range(50):
            if called["count"] >= 2:
                break
            time.sleep(0.05)
        assert called["count"] >= 2

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
    assert trainer._thread is not None
    assert not trainer._thread.is_alive()
