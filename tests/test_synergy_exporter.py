import importlib
import json
import socket
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


def test_exporter_serves_latest_metrics(tmp_path: Path) -> None:
    se = importlib.import_module("menace.synergy_exporter")
    me = importlib.import_module("menace.metrics_exporter")

    history = [
        {"synergy_roi": 0.2, "synergy_efficiency": 0.4},
        {"synergy_roi": 0.3, "synergy_efficiency": 0.5},
    ]
    hist_file = tmp_path / "synergy_history.json"
    hist_file.write_text(json.dumps(history))

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)
    exp.start()
    try:
        metrics = {}
        for _ in range(50):
            try:
                data = urllib.request.urlopen(
                    f"http://localhost:{port}/metrics"
                ).read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.3:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.3
        assert metrics.get("synergy_efficiency") == 0.5
    finally:
        exp.stop()
        me.stop_metrics_server()

    assert exp._thread is not None
    assert not exp._thread.is_alive()
