import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import subprocess  # noqa: E402
import os  # noqa: E402
import menace.advanced_error_management as aem  # noqa: E402
import menace.watchdog as wd  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
import menace.error_bot as eb  # noqa: E402
import menace.resource_allocation_optimizer as rao  # noqa: E402
import menace.data_bot as db  # noqa: E402
from menace.error_logger import TelemetryEvent  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from dynamic_path_router import resolve_path  # noqa: E402


def _setup_dbs(tmp_path: Path):
    _ = resolve_path("sandbox_runner.py")  # path-ignore
    err = eb.ErrorDB(tmp_path / "e.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    return err, roi, metrics


def test_formal_verifier(tmp_path, monkeypatch):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("x = 1\n")

    calls = []

    def fake_run(cmd, capture_output=True, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    verifier = aem.FormalVerifier()
    assert verifier.verify(path)
    assert calls


def test_secure_log(tmp_path):
    log = aem.SecureLog(tmp_path / "log.txt")
    log.append("line")
    assert log.hashes
    assert log.path.exists()


def test_secure_log_export(tmp_path):
    log = aem.SecureLog(tmp_path / "log.txt")
    log.append("a")
    log.append("b")
    out = tmp_path / "out.txt"
    log.export(out)
    assert out.read_text().splitlines() == ["a", "b"]


def test_secure_log_export_detects_tampering(tmp_path):
    log = aem.SecureLog(tmp_path / "log.txt")
    log.append("ok")
    tampered = log.path.read_text().replace("ok", "bad")
    log.path.write_text(tampered)
    with pytest.raises(RuntimeError):
        log.export(tmp_path / "out.txt")


def test_playbook_generator(tmp_path):
    gen = aem.PlaybookGenerator()
    pb = gen.generate(["issue"])
    assert os.path.exists(pb)


def test_compile_dossier_generates_playbook(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    err_db.add_telemetry(TelemetryEvent(stack_trace="boom"))
    roi_db.add(rao.KPIRecord(bot="b", revenue=100.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    roi_db.add(rao.KPIRecord(bot="b", revenue=80.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    old_ts = (datetime.utcnow() - timedelta(hours=3)).isoformat()
    metrics_db.add(db.MetricRecord(bot="b", cpu=90.0, memory=90.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=1, ts=old_ts))  # noqa: E501

    called = []

    def fake_generate(self, steps):
        called.append(steps)
        path = tmp_path / "pb.json"
        path.write_text("[]")
        return str(path)

    monkeypatch.setattr(aem.PlaybookGenerator, "generate", fake_generate)
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(err_db, roi_db, metrics_db, context_builder=builder)
    dossier, attachments = watch.compile_dossier()
    assert any(str(tmp_path / "pb.json") == a for a in attachments)
    assert called


def test_watchdog_notifications_include_playbook(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    for _ in range(4):
        err_db.add_telemetry(TelemetryEvent(stack_trace="boom"))
    roi_db.add(rao.KPIRecord(bot="b", revenue=100.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    roi_db.add(rao.KPIRecord(bot="b", revenue=80.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    old_ts = (datetime.utcnow() - timedelta(hours=3)).isoformat()
    metrics_db.add(db.MetricRecord(bot="b", cpu=90.0, memory=90.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=1, ts=old_ts))  # noqa: E501

    pb = tmp_path / "pb.json"

    def fake_generate(self, steps):
        pb.write_text("[]")
        return str(pb)

    monkeypatch.setattr(aem.PlaybookGenerator, "generate", fake_generate)
    sent = {}

    def fake_notify(msg, attachments=None):
        sent["attachments"] = attachments

    notifier = wd.Notifier()
    notifier.notify = fake_notify
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db, roi_db, metrics_db, notifier=notifier, context_builder=builder
    )
    watch.check()
    assert any(str(pb) == a for a in sent.get("attachments", []))


def test_formal_verifier_symbolic_detects_exception(tmp_path, monkeypatch):
    if aem.angr is None:
        pytest.skip("angr not installed")
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text(
        "def foo(x):\n    if x:\n        raise ValueError('boom')\n    return 1\n"
    )

    def fake_run(cmd, capture_output=True, check=False):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    verifier = aem.FormalVerifier()
    assert "symbolic" in verifier.tools
    assert not verifier.verify(mod)
