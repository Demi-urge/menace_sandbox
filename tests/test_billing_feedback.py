import json
import sys
import types


class DummyMemory:
    def __init__(self):
        self.entries = []

    def log_interaction(self, instruction, metadata_json, tags=None):
        self.entries.append((instruction, metadata_json, tags))


MEMORY = DummyMemory()

# Provide a lightweight menace_sanity_layer stub before importing stripe_watchdog
sanity_stub = types.ModuleType("menace_sanity_layer")
sanity_stub.EVENT_TYPE_INSTRUCTIONS = {
    "missing_charge": (
        "Avoid creating Stripe charges without billing log entries or central routing."
    ),
}


def record_billing_event(
    event_type, metadata, instruction, *, config_path=None, self_coding_engine=None
):
    MEMORY.log_interaction(
        instruction,
        json.dumps(
            {"event_type": event_type, "metadata": metadata}, sort_keys=True
        ),
        tags=["billing"],
    )
    if self_coding_engine is not None:
        update = getattr(self_coding_engine, "update_generation_params", None)
        if callable(update):
            update(metadata)


sanity_stub.record_billing_event = record_billing_event
sanity_stub.record_event = lambda *a, **k: None
sanity_stub.record_billing_anomaly = lambda *a, **k: None
sanity_stub.record_payment_anomaly = lambda *a, **k: None
sanity_stub.fetch_recent_billing_issues = lambda *a, **k: []
sanity_stub.refresh_billing_instructions = lambda *a, **k: None

sys.modules["menace_sanity_layer"] = sanity_stub

import stripe_watchdog  # noqa: E402  (import after stubbing)


class DummyEngine:
    def __init__(self):
        self.calls = []

    def update_generation_params(self, metadata):
        self.calls.append(metadata)
        return {}


def test_emit_anomaly_triggers_feedback(monkeypatch, tmp_path):
    engine = DummyEngine()

    # Avoid touching external systems
    monkeypatch.setattr(stripe_watchdog, "_refresh_instruction_cache", lambda: None)
    monkeypatch.setattr(stripe_watchdog, "load_api_key", lambda: None)
    monkeypatch.setattr(stripe_watchdog.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(stripe_watchdog, "CONFIG_PATH", tmp_path / "cfg.json")

    record = {"type": "missing_charge", "id": "ch_123", "stripe_account": "acct_1"}

    builder = types.SimpleNamespace(build=lambda *a, **k: "")
    stripe_watchdog._emit_anomaly(
        record, False, False, self_coding_engine=engine, context_builder=builder
    )

    # Instruction should be logged to GPT memory
    assert MEMORY.entries, "instruction not logged"
    instruction, metadata_json, tags = MEMORY.entries[0]
    assert "Avoid" in instruction
    metadata = json.loads(metadata_json)
    assert metadata["metadata"]["id"] == "ch_123"

    # Self-coding engine receives anomaly metadata
    assert engine.calls, "update_generation_params not invoked"
    assert engine.calls[0]["id"] == "ch_123"
