import logging
import subprocess
import menace.security_auditor as sa


def test_audit_failure_logs(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, capture_output=True, text=True, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    caplog.set_level(logging.ERROR)
    audit = sa.SecurityAuditor()
    assert not audit.audit()
    assert calls and calls[0][0] == "bandit"
    assert "failed" in caplog.text
