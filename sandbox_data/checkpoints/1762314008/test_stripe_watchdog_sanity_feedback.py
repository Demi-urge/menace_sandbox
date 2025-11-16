import builtins
import importlib
import logging
import sys
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helper to reload stripe_watchdog with menace_sanity_layer import failing
# ---------------------------------------------------------------------------


def _reload_watchdog_with_missing_sanity_layer(monkeypatch):
    """Reload stripe_watchdog simulating menace_sanity_layer import failure."""
    monkeypatch.delitem(sys.modules, "stripe_watchdog", raising=False)
    monkeypatch.delitem(sys.modules, "menace_sanity_layer", raising=False)
    sys.modules.setdefault("vector_service", SimpleNamespace(CognitionLayer=lambda: None))

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - helper
        if name == "menace_sanity_layer":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("stripe_watchdog")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_emit_anomaly_records_billing_feedback(monkeypatch):
    sys.modules.setdefault("vector_service", SimpleNamespace(CognitionLayer=lambda: None))
    import stripe_watchdog as sw

    billing_calls = []
    event_calls = []
    payment_calls = []

    def fake_billing(event_type, metadata, *, severity=1.0, **kwargs):
        billing_calls.append((event_type, metadata, severity))

    def fake_event(event_type, metadata, instruction, **kwargs):
        event_calls.append((event_type, metadata, instruction))

    def fake_payment(event_type, metadata, instruction, **kwargs):
        payment_calls.append((event_type, metadata, instruction))

    monkeypatch.setattr(sw, "record_billing_anomaly", fake_billing)
    monkeypatch.setattr(sw, "record_billing_event", fake_event)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", fake_payment)
    monkeypatch.setattr(sw, "record_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", SimpleNamespace(record=lambda *a, **k: None))
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)

    record = {"type": "missing_charge", "id": "ch_1", "stripe_account": "acct_1"}
    builder = SimpleNamespace(build=lambda *a, **k: "")
    sw._emit_anomaly(record, False, False, context_builder=builder)

    assert billing_calls and billing_calls[0][0] == "missing_charge"
    assert billing_calls[0][1]["id"] == "ch_1"
    assert billing_calls[0][1]["stripe_account"] == "acct_1"
    assert billing_calls[0][2] == sw.SEVERITY_MAP["missing_charge"]

    assert event_calls and event_calls[0][0] == "missing_charge"
    meta = event_calls[0][1]
    assert meta["id"] == "ch_1" and meta["stripe_account"] == "acct_1"
    assert "timestamp" in meta
    expected_instruction = sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    assert event_calls[0][2] == expected_instruction

    assert payment_calls and payment_calls[0][0] == "missing_charge"
    pmeta = payment_calls[0][1]
    assert pmeta["charge_id"] == "ch_1"
    assert pmeta["reason"] == "missing_charge"
    assert payment_calls[0][2] == expected_instruction


def test_warning_when_sanity_layer_missing(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _reload_watchdog_with_missing_sanity_layer(monkeypatch)
    assert "menace_sanity_layer import failed" in caplog.text
