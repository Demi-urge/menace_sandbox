"""Tests for :mod:`security_auditor` utilities."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

# Ensure the package root is importable when tests are executed from the
# repository directory.
sys.path.append(str(Path(__file__).resolve().parents[2]))

import menace_sandbox.security_auditor as security_auditor
import menace_sandbox.alignment_review_agent as alignment_review_agent
from menace_sandbox.alignment_review_agent import AlignmentReviewAgent


def test_dispatch_alignment_warning_uses_audit(monkeypatch):
    """Ensure dispatch function forwards records to the auditor."""

    recorded: list[Mapping[str, Any]] = []

    class DummyAuditor:
        def audit(self, report: Mapping[str, Any]) -> bool:  # pragma: no cover - simple
            recorded.append(report)
            return True

    dummy = DummyAuditor()
    monkeypatch.setattr(security_auditor, "_AUDITOR", dummy)

    report = {"entry_id": "123"}
    security_auditor.dispatch_alignment_warning(report)

    assert recorded == [report]


def test_alignment_review_agent_invokes_auditor(monkeypatch):
    """The review agent should audit each newly seen warning."""

    calls: list[Mapping[str, Any]] = []

    class DummyAuditor:
        def audit(self, report: Mapping[str, Any]) -> bool:  # pragma: no cover - simple
            calls.append(report)
            return True

    auditor = DummyAuditor()

    warn1 = {"entry_id": "1"}
    warn2 = {"entry_id": "2"}
    warn3 = {"entry_id": "3"}
    sequence = [[warn1, warn2], [warn1, warn2, warn3]]

    def fake_load() -> list[Mapping[str, Any]]:
        data = sequence.pop(0)
        if not sequence:
            agent._stop.set()
        return data

    agent = AlignmentReviewAgent(interval=0, auditor=auditor)
    monkeypatch.setattr(alignment_review_agent, "load_recent_alignment_warnings", fake_load)

    agent._run()  # run the loop once per sequence entry

    assert "summary" in calls[0]
    filtered = [c for c in calls if "summary" not in c]
    assert filtered == [warn1, warn2, warn3]


