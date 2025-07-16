import os
import sys
import types
import signal

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub cryptography for AuditTrail dependency
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

import menace.compliance_audit_service as cas


def test_log_parsing_and_callback(monkeypatch):
    records = []
    svc = cas.ComplianceAuditService(on_violation=lambda c: records.append(c), test_mode=True)
    monkeypatch.setattr(svc.auditor, "audit", lambda: True)
    svc._test_log_lines = [
        "sig1 {\"type\": \"volume\"}",
        "bad line",
        "  sig2   {\"type\": \"permission\"}  ",
        "sig3 {\"type\": \"volume\"} trailing",
    ]
    svc._run_once()
    assert records == [{"volume": 1, "permission": 1}]


def test_run_continuous_uses_env(monkeypatch):
    monkeypatch.setattr(cas, "BackgroundScheduler", None)
    added = {}

    def fake_add_job(self, func, interval, id):
        added["interval"] = interval
        added["id"] = id

    monkeypatch.setattr(cas._SimpleScheduler, "add_job", fake_add_job)
    monkeypatch.setenv("AUDIT_INTERVAL", "123")

    called_signals = []
    monkeypatch.setattr(cas.signal, "signal", lambda sig, handler: called_signals.append(sig))

    svc = cas.ComplianceAuditService(test_mode=True)
    svc.run_continuous()
    assert added["interval"] == 123
    assert called_signals == []


def test_retry_config(monkeypatch):
    monkeypatch.setenv("AUDIT_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("AUDIT_RETRY_DELAY", "2.5")

    captured = {}

    def fake_with_retry(func, *, attempts, delay, logger, exc=Exception):
        captured["attempts"] = attempts
        captured["delay"] = delay
        return func()

    monkeypatch.setattr(cas, "with_retry", fake_with_retry)

    svc = cas.ComplianceAuditService(test_mode=True)
    monkeypatch.setattr(svc.auditor, "audit", lambda: True)
    svc._test_log_lines = []
    svc._run_once()

    assert captured["attempts"] == 5
    assert captured["delay"] == 2.5

