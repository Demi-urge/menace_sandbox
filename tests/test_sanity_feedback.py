
def test_anomaly_triggers_self_coding_update(monkeypatch):
    import stripe_watchdog as sw
    import menace_sanity_layer as msl

    # Enable sanity layer feedback and silence IO
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: None)

    calls: list[dict] = []

    class DummyEngine:
        def update_generation_params(self, metadata):
            calls.append(metadata)

    class DummyTelemetry:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []
            self.checked = False

        def record_event(self, event_type, metadata):
            self.events.append((event_type, metadata))

        def check(self):
            self.checked = True

    engine = DummyEngine()
    telemetry = DummyTelemetry()

    charges = [{"id": "ch_1", "amount": 5, "receipt_email": "a@example.com", "created": 1}]

    sw.detect_missing_charges(
        charges,
        [],
        [],
        write_codex=False,
        export_training=False,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )

    assert calls, "SelfCodingEngine was not updated"
    assert telemetry.events and telemetry.checked
