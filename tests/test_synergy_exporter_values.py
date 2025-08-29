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


def test_metrics_and_health_update(tmp_path: Path) -> None:
    se = importlib.import_module("menace.synergy_exporter")
    db_mod = importlib.import_module("menace.synergy_history_db")
    me = importlib.import_module("menace.metrics_exporter")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    conn.close()

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)
    exp.start()
    try:
        conn = db_mod.connect(hist_file)
        db_mod.insert_entry(conn, {"synergy_roi": 0.1, "security_score": 50})
        conn.close()

        metrics: dict[str, float] = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.1 and metrics.get("security_score") == 50:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.1
        assert metrics.get("security_score") == 50

        health = urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health"
        ).read().decode()
        info = json.loads(health)
        assert info["healthy"] is True
        first_ts = info.get("last_update")
        assert isinstance(first_ts, float)

        conn = db_mod.connect(hist_file)
        db_mod.insert_entry(conn, {"synergy_roi": 0.2, "security_score": 60})
        conn.close()

        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.2 and metrics.get("security_score") == 60:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.2
        assert metrics.get("security_score") == 60

        health2 = urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health"
        ).read().decode()
        info2 = json.loads(health2)
        assert info2["healthy"] is True
        second_ts = info2.get("last_update")
        assert isinstance(second_ts, float)
        assert second_ts >= first_ts
    finally:
        exp.stop()
        me.stop_metrics_server()
