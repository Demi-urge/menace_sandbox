"""Tests for the background alignment review agent."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping
from unittest.mock import MagicMock

# Ensure the package root is importable when tests are executed from the
# repository directory.
sys.path.append(str(Path(__file__).resolve().parents[2]))

import menace_sandbox.alignment_review_agent as alignment_review_agent
from menace_sandbox.alignment_review_agent import AlignmentReviewAgent
from menace_sandbox.security_auditor import SecurityAuditor


def test_agent_audits_each_new_warning_once(monkeypatch):
    """Audit is invoked once per new warning and repeats are ignored."""
    warnings = [{"entry_id": "1"}, {"entry_id": "2"}]

    audit_mock = MagicMock()
    monkeypatch.setattr(SecurityAuditor, "audit", audit_mock)

    agent = AlignmentReviewAgent(interval=0)

    polls = 0

    def fake_load() -> list[Mapping[str, str]]:
        nonlocal polls
        polls += 1
        if polls >= 2:
            agent._stop.set()
        return warnings

    monkeypatch.setattr(
        alignment_review_agent, "load_recent_alignment_warnings", fake_load
    )

    agent._run()

    assert audit_mock.call_count == len(warnings)
    audited = [c.args[0] for c in audit_mock.call_args_list]
    assert audited == warnings
    assert polls >= 2
