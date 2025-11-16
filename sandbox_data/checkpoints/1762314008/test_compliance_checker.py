import importlib.util
import pytest

if importlib.util.find_spec("cryptography") is None:
    pytest.skip("optional dependencies not installed", allow_module_level=True)

import menace.compliance_checker as cc


def test_compliance_violation_logged(monkeypatch):
    records = []

    class DummyTrail:
        def __init__(self, path):
            pass

        def record(self, msg):
            records.append(msg)

    monkeypatch.setattr(cc, "AuditTrail", DummyTrail)
    checker = cc.ComplianceChecker()
    assert not checker.check_trade({"volume": 2001})
    assert records
    assert not checker.verify_permission("viewer", "trade")
    assert any(r.get("type") == "permission" and not r.get("allowed") for r in records)

