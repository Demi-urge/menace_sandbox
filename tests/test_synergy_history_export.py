import json
import socket
import time
import urllib.request
import importlib
import importlib.util
import sys
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


def test_history_persistence_and_export(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    se_spec = importlib.util.spec_from_file_location(
        "sandbox_runner.synergy_exporter", root / "synergy_exporter.py"
    )
    se = importlib.util.module_from_spec(se_spec)
    me_spec = importlib.util.spec_from_file_location(
        "sandbox_runner.metrics_exporter", root / "metrics_exporter.py"
    )
    me_alias = importlib.util.module_from_spec(me_spec)
    me_spec.loader.exec_module(me_alias)
    sys.modules.setdefault("sandbox_runner.metrics_exporter", me_alias)
    se_spec.loader.exec_module(se)

    db_mod = importlib.import_module("menace.synergy_history_db")
    me = importlib.import_module("menace.metrics_exporter")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    conn.close()

    port = _free_port()
    exporter = se.start_synergy_exporter(history_file=hist_file, interval=0.05, port=port)
    try:
        db_mod.record(hist_file, {"synergy_roi": 0.1})
        metrics = {}
        for _ in range(40):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.1:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.1
        assert se.exporter_failures.labels().get() == 0.0
        assert se.exporter_uptime.labels().get() >= 0.0

        db_mod.record(hist_file, {"synergy_roi": 0.2})
        for _ in range(40):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.2:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.2

        health = urllib.request.urlopen(
            f"http://localhost:{exporter.health_port}/health"
        ).read().decode()
        info = json.loads(health)
        assert info["status"] == "ok"
    finally:
        exporter.stop()
        me.stop_metrics_server()

    history = db_mod.load_history(hist_file)
    assert history == [{"synergy_roi": 0.1}, {"synergy_roi": 0.2}]
    assert exporter._thread is not None
    assert not exporter._thread.is_alive()
    assert se.exporter_failures.labels().get() == 0.0
    assert se.exporter_uptime.labels().get() > 0
