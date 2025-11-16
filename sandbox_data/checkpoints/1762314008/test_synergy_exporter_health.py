import json
import socket
import time
import urllib.request
from pathlib import Path

import importlib


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


def test_health_endpoint_returns_json(tmp_path: Path) -> None:
    se = importlib.import_module("menace.synergy_exporter")
    db_mod = importlib.import_module("menace.synergy_history_db")
    me = importlib.import_module("menace.metrics_exporter")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.42})
    conn.close()

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)
    exp.start()
    try:
        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.42:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.42

        health = urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health"
        ).read().decode()
        info = json.loads(health)
        assert info["healthy"] is True
        assert isinstance(info.get("last_update"), float)
    finally:
        exp.stop()
        me.stop_metrics_server()
