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
from menace_sandbox.alignment_review_agent import (
    AlignmentReviewAgent,
    summarize_warnings,
)
from menace_sandbox.security_auditor import SecurityAuditor


def test_agent_emits_summary_and_records(tmp_path, monkeypatch):
    """Warnings trigger a summary followed by individual audit calls."""

    log_path = tmp_path / "violation_log.jsonl"
    db_path = tmp_path / "alignment_warnings.db"
    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(violation_logger, "LOG_PATH", str(log_path))
    monkeypatch.setattr(violation_logger, "ALIGNMENT_DB_PATH", str(db_path))

    entry_id = "warn-123"
    violation_logger.log_violation(
        entry_id,
        "alignment",
        4,
        {"detail": "test", "patch_link": "mod1"},
        alignment_warning=True,
    )

    auditor = MagicMock(spec=SecurityAuditor)
    agent = AlignmentReviewAgent(interval=0.01, auditor=auditor)
    agent.start()
    try:
        timeout = time.time() + 2
        while time.time() < timeout and auditor.audit.call_count < 2:
            time.sleep(0.05)
    finally:
        agent.stop()

    assert auditor.audit.call_count >= 2
    summary_call = auditor.audit.call_args_list[0][0][0]
    record_call = auditor.audit.call_args_list[1][0][0]
    assert "summary" in summary_call
    assert record_call["entry_id"] == entry_id


def test_summarize_warnings_counts_and_modules(tmp_path, monkeypatch):
    """Summaries aggregate counts and modules from pending warnings."""

    log_path = tmp_path / "violation_log.jsonl"
    db_path = tmp_path / "alignment_warnings.db"
    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(violation_logger, "LOG_PATH", str(log_path))
    monkeypatch.setattr(violation_logger, "ALIGNMENT_DB_PATH", str(db_path))

    violation_logger.log_violation(
        "w1",
        "alignment",
        2,
        {"patch_link": "module_a"},
        alignment_warning=True,
    )
    violation_logger.log_violation(
        "w2",
        "alignment",
        5,
        {"patch_link": "module_b"},
        alignment_warning=True,
    )

    summary = summarize_warnings()
    assert summary["counts"][2] == 1
    assert summary["counts"][5] == 1
    assert set(summary["modules"]) == {"module_a", "module_b"}


def test_cli_summary_outputs(tmp_path, monkeypatch, capsys):
    """CLI ``--summary`` flag prints the latest summary."""

    log_path = tmp_path / "violation_log.jsonl"
    db_path = tmp_path / "alignment_warnings.db"
    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(violation_logger, "LOG_PATH", str(log_path))
    monkeypatch.setattr(violation_logger, "ALIGNMENT_DB_PATH", str(db_path))

    violation_logger.log_violation(
        "w1",
        "alignment",
        3,
        {"patch_link": "module_c"},
        alignment_warning=True,
    )

    import menace_sandbox.alignment_review_agent as ara

    monkeypatch.setattr(
        sys,
        "argv",
        ["alignment_review_agent", "--summary"],
    )
    ara._cli()
    out = capsys.readouterr().out
    assert "module_c" in out
