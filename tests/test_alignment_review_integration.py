from __future__ import annotations

import sys
import time
import types
from pathlib import Path

# Ensure package root importable
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dynamic_path_router import resolve_path  # noqa: E402
import menace_sandbox.self_improvement as sie  # noqa: E402
import menace_sandbox.human_alignment_agent as haa  # noqa: E402
import menace_sandbox.violation_logger as violation_logger  # noqa: E402
import menace_sandbox.alignment_review_agent as ara  # noqa: E402
import menace_sandbox.security_auditor as security_auditor  # noqa: E402
from sandbox_settings import SandboxSettings  # noqa: E402


def test_alignment_review_integration(tmp_path, monkeypatch):
    """End-to-end check from improvement warning to security audit."""

    log_path = tmp_path / "violation_log.jsonl"
    db_path = tmp_path / "alignment_warnings.db"
    _ = resolve_path("sandbox_runner.py")  # path-ignore
    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(violation_logger, "LOG_PATH", str(log_path))
    monkeypatch.setattr(violation_logger, "ALIGNMENT_DB_PATH", str(db_path))

    logged: list[tuple[tuple, dict]] = []
    orig_log = violation_logger.log_violation

    def capture_log(*args, **kwargs):
        logged.append((args, kwargs))
        orig_log(*args, **kwargs)

    monkeypatch.setattr(violation_logger, "log_violation", capture_log)
    monkeypatch.setattr(haa, "log_violation", capture_log)

    metrics = {"accuracy": 0.95, "previous_accuracy": 0.9}
    changes = [{"file": "mod.py", "code": "def f():\n    eval('2+2')\n"}]  # path-ignore
    settings = SandboxSettings(improvement_warning_threshold=0.0)
    agent = sie.HumanAlignmentAgent(settings=settings)
    warnings = agent.evaluate_changes(changes, metrics, [])

    assert logged, "HumanAlignmentAgent did not log a warning"
    entry_id = logged[0][0][0]

    engine = types.SimpleNamespace(
        warning_summary=[],
        logger=types.SimpleNamespace(warning=lambda *a, **k: None),
    )
    sie.SelfImprovementEngine._record_warning_summary(engine, 0.1, warnings)

    persisted = violation_logger.load_persisted_alignment_warnings()
    assert any(rec["entry_id"] == entry_id for rec in persisted)

    retrieved: list[dict] = []
    orig_load = violation_logger.load_persisted_alignment_warnings

    def capture_load(*args, **kwargs):
        recs = orig_load(*args, **kwargs)
        retrieved.extend(recs)
        return recs

    monkeypatch.setattr(violation_logger, "load_persisted_alignment_warnings", capture_load)
    monkeypatch.setattr(ara, "load_persisted_alignment_warnings", capture_load)

    dispatched: list[dict] = []

    def fake_dispatch(record):
        dispatched.append(record)

    monkeypatch.setattr(security_auditor, "dispatch_alignment_warning", fake_dispatch)

    auditor = security_auditor.SecurityAuditor(
        escalate_hook=security_auditor.dispatch_alignment_warning
    )
    review_agent = ara.AlignmentReviewAgent(interval=0.01, auditor=auditor)
    review_agent.start()
    try:
        timeout = time.time() + 2
        while time.time() < timeout and not dispatched:
            time.sleep(0.05)
    finally:
        review_agent.stop()

    assert any(rec["entry_id"] == entry_id for rec in retrieved)
    assert any(rec.get("entry_id") == entry_id for rec in dispatched)
