
def test_anomaly_triggers_self_coding_update(monkeypatch):
    import types
    import sys
    sys.modules.setdefault(
        "vector_service",
        types.SimpleNamespace(
            CognitionLayer=object, PatchLogger=object, VectorServiceError=Exception
        ),
    )
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


def test_generation_param_persistence(monkeypatch, tmp_path):
    import types
    import sys
    import json
    sys.modules.setdefault(
        "vector_service",
        types.SimpleNamespace(
            CognitionLayer=object, PatchLogger=object, VectorServiceError=Exception
        ),
    )
    import menace_sanity_layer as msl

    # Stub memory and auditing to avoid side effects
    mm = types.SimpleNamespace(query=lambda *a, **k: [], store=lambda *a, **k: None)
    monkeypatch.setattr(msl, "_get_memory_manager", lambda: mm)
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)

    state_file = tmp_path / "state.json"

    class DummyEngine:
        def __init__(self):
            self.params: dict[str, object] = {}

        def update_generation_params(self, meta):
            self.params.update(meta)
            state_file.write_text(json.dumps(self.params))
            return meta

    engine = DummyEngine()
    # Trigger correction on first anomaly
    msl.ANOMALY_THRESHOLDS["missing_charge"] = 1
    msl.ANOMALY_HINTS["missing_charge"] = {"block_unlogged_charges": True}

    msl.record_payment_anomaly(
        "missing_charge", {"charge_id": "c1"}, self_coding_engine=engine
    )

    data = json.loads(state_file.read_text())
    assert engine.params.get("block_unlogged_charges") is True
    assert data.get("block_unlogged_charges") is True
