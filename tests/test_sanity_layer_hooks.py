import json
import sys
import types


sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(CognitionLayer=lambda: None)
)

import menace_sanity_layer as msl  # noqa: E402


def test_emit_anomaly_triggers_record_event(monkeypatch):
    import stripe_watchdog as sw

    calls = []

    def fake_record_event(event_type, metadata):
        calls.append((event_type, metadata))

    monkeypatch.setattr(msl, "record_event", fake_record_event)
    monkeypatch.setattr(sw, "record_event", fake_record_event)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)

    sw._emit_anomaly({"type": "missing_charge", "charge_id": "ch_1"}, False, False)

    assert calls == [("missing_charge", {"charge_id": "ch_1"})]


def _stub_unified_event_bus(monkeypatch, tmp_path):
    import dynamic_path_router
    from dynamic_path_router import resolve_path as _orig_resolve

    stub_path = tmp_path / "unified_event_bus.py"
    stub_path.write_text("class UnifiedEventBus:\n    pass\n")

    def fake_resolve(name, root=None):
        if name == "unified_event_bus.py":
            return stub_path
        try:
            return _orig_resolve(name, root)
        except TypeError:
            return _orig_resolve(name)

    monkeypatch.setattr(dynamic_path_router, "resolve_path", fake_resolve)


def test_alert_mismatch_invokes_record_event(monkeypatch, tmp_path):
    from tests.test_stripe_billing_router_logging import _import_module

    _stub_unified_event_bus(monkeypatch, tmp_path)
    sbr = _import_module(monkeypatch, tmp_path)

    calls: list[tuple[str, dict]] = []

    def fake_record_event(event_type, metadata):
        calls.append((event_type, metadata))

    monkeypatch.setattr(msl, "record_event", fake_record_event)
    monkeypatch.setattr(sbr, "record_payment_anomaly", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_critical_discrepancy", lambda *a, **k: None)
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    import evolution_lock_flag

    monkeypatch.setattr(evolution_lock_flag, "trigger_lock", lambda *a, **k: None)

    sbr._alert_mismatch("bot1", "acct_mismatch")

    assert calls == [
        (
            "account_mismatch",
            {"bot_id": "bot1", "destination_account": "acct_mismatch", "amount": None},
        )
    ]


def test_record_event_logs_instruction_and_tags(monkeypatch):
    calls = []

    class DummyMemory:
        def log_interaction(self, instruction, content, *, tags=None):
            calls.append((instruction, json.loads(content), tags or []))

    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())

    msl.record_event("missing_charge", {"charge_id": "ch_2"})

    assert calls
    instruction, payload, tags = calls[0]
    assert "Avoid generating bots that make Stripe charges" in instruction
    assert payload == {
        "event_type": "missing_charge",
        "metadata": {"charge_id": "ch_2"},
    }
    assert msl.FEEDBACK in tags and msl.ERROR_FIX in tags and "missing_charge" in tags
