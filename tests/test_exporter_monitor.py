import importlib
import socket
import time
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
    from pathlib import Path

    setup_stubs(monkeypatch, tmp_path)

    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"
    sys.modules.pop("run_autonomous", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    return mod


def test_monitor_restarts_exporter(monkeypatch, tmp_path: Path) -> None:
    ra = _load_run_autonomous(monkeypatch, tmp_path)
    se = importlib.import_module("menace.synergy_exporter")
    db_mod = importlib.import_module("menace.synergy_history_db")
    me = importlib.import_module("menace.metrics_exporter")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.55})
    conn.close()

    port = _free_port()
    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=port)
    exp.start()

    log = ra.AuditTrail(str(tmp_path / "trail.log"))
    mon = ra.ExporterMonitor(exp, log, interval=0.05)
    mon.start()
    try:
        for _ in range(40):
            if ra._exporter_health_ok(exp):
                break
            time.sleep(0.05)
        assert ra._exporter_health_ok(exp)

        exp.stop()
        for _ in range(60):
            if mon.restart_count > 0 and ra._exporter_health_ok(mon.exporter):
                break
            time.sleep(0.05)

        assert mon.restart_count >= 1
        assert ra._exporter_health_ok(mon.exporter)

        metrics = {}
        for _ in range(40):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_roi") == 0.55:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        assert metrics.get("synergy_roi") == 0.55
    finally:
        mon.stop()
        me.stop_metrics_server()
