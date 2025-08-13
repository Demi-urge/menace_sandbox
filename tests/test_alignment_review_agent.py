"""Tests for the background alignment review agent."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the package root is importable when tests are executed from the
# repository directory.
sys.path.append(str(Path(__file__).resolve().parents[2]))

import menace_sandbox.violation_logger as violation_logger
from menace_sandbox.alignment_review_agent import AlignmentReviewAgent
from menace_sandbox.security_auditor import SecurityAuditor


def test_agent_forwards_logged_warning(tmp_path, monkeypatch):
    """A logged alignment warning is passed to ``SecurityAuditor.audit``."""

    log_path = tmp_path / "violation_log.jsonl"
    db_path = tmp_path / "alignment_warnings.db"
    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(violation_logger, "LOG_PATH", str(log_path))
    monkeypatch.setattr(violation_logger, "ALIGNMENT_DB_PATH", str(db_path))

    entry_id = "warn-123"
    violation_logger.log_violation(
        entry_id, "alignment", 4, {"detail": "test"}, alignment_warning=True
    )

    auditor = MagicMock(spec=SecurityAuditor)
    agent = AlignmentReviewAgent(interval=0.01, auditor=auditor)
    agent.start()
    try:
        timeout = time.time() + 2
        while time.time() < timeout and auditor.audit.call_count == 0:
            time.sleep(0.05)
    finally:
        agent.stop()

    auditor.audit.assert_called_once()
    record = auditor.audit.call_args[0][0]
    assert record["entry_id"] == entry_id
