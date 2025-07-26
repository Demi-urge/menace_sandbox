import importlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path
import multiprocessing


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


def _worker(data_dir: str, port: int, call_log: str) -> None:
    os.environ["EXPORT_SYNERGY_METRICS"] = "1"
    os.environ["AUTO_TRAIN_SYNERGY"] = "1"
    os.environ["AUTO_TRAIN_INTERVAL"] = "0.05"
    os.environ["SYNERGY_METRICS_PORT"] = str(port)
    os.environ["SYNERGY_EXPORTER_CHECK_INTERVAL"] = "0.05"
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    os.chdir(data_dir)

    import synergy_tools as tools
    import menace.synergy_auto_trainer as sat
    import menace.synergy_exporter as se
    import synergy_monitor as sym_mon
    import types

    class DummyTrail:
        def __init__(self, path, private_key=None) -> None:
            self.path = path
        def record(self, message) -> None:
            pass

    at_mod = types.ModuleType("menace.audit_trail")
    at_mod.AuditTrail = DummyTrail
    sys.modules["menace.audit_trail"] = at_mod

    import json

    def fake_train(history, path):
        with open(call_log, "a") as fh:
            json.dump({"history": list(history), "path": str(path)}, fh)
            fh.write("\n")
        return {}

    sat.synergy_weight_cli.train_from_history = fake_train

    sym_mon.synergy_exporter_restarts_total.set(0.0)
    called = {"done": False}

    orig_health = sym_mon._exporter_health_ok

    def fake_health(exp, *, max_age: float = 30.0):
        if not called["done"]:
            called["done"] = True
            return False
        return orig_health(exp, max_age=max_age)

    sym_mon._exporter_health_ok = fake_health

    class FastExporter(se.SynergyExporter):
        def __init__(self, history_file, port):
            super().__init__(history_file, interval=0.05, port=port)
    tools.SynergyExporter = FastExporter

    tools.main(["--sandbox-data-dir", data_dir])


def test_synergy_tools_services(tmp_path: Path) -> None:
    db_mod = importlib.import_module("menace.synergy_history_db")

    hist_file = tmp_path / "synergy_history.db"
    conn = db_mod.connect(hist_file)
    db_mod.insert_entry(conn, {"synergy_roi": 0.5})
    conn.close()

    weights_file = tmp_path / "synergy_weights.json"
    weights_file.write_text("{}")

    port = _free_port()
    call_log = tmp_path / "trainer_calls.txt"

    proc = multiprocessing.Process(
        target=_worker,
        args=(str(tmp_path), port, str(call_log)),
    )
    proc.start()
    try:
        metrics = {}
        for _ in range(70):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if (
                    metrics.get("synergy_exporter_uptime_seconds", 0.0) > 0
                    and metrics.get("synergy_trainer_iterations", 0.0) >= 1
                ):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        assert metrics.get("synergy_exporter_uptime_seconds", 0.0) > 0
        assert metrics.get("synergy_trainer_iterations", 0.0) >= 1

        for _ in range(50):
            try:
                data = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
                metrics = _parse_metrics(data)
                if metrics.get("synergy_exporter_restarts_total", 0.0) >= 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        assert metrics.get("synergy_exporter_restarts_total", 0.0) >= 1
    finally:
        os.kill(proc.pid, signal.SIGINT)
        proc.join(5)

    assert proc.exitcode == 0
    assert call_log.exists()
    data = json.loads(call_log.read_text().splitlines()[0])
    assert Path(data["path"]) == tmp_path / "synergy_weights.json"
    assert data["history"][0]["synergy_roi"] == 0.5
