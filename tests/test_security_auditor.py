import logging
import subprocess
import menace.security_auditor as sa


def test_audit_disabled(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, capture_output=True, text=True, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    caplog.set_level(logging.INFO)
    audit = sa.SecurityAuditor()
    assert audit.audit()
    assert not calls
    assert "disabled" in caplog.text
