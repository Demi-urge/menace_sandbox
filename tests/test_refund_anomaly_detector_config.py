import json
import os
from types import SimpleNamespace

from dynamic_path_router import resolve_path

import menace_sanity_layer as msl
from billing import refund_anomaly_detector as rad


def _make_event(event_id: str, event_type: str, bot_id: str, *, created: int) -> SimpleNamespace:
    data_obj = {
        "metadata": {"bot_id": bot_id},
        "amount": 100,
        "currency": "usd",
        "receipt_email": "user@example.com",
    }
    return SimpleNamespace(
        id=event_id,
        type=event_type,
        created=created,
        data=SimpleNamespace(object=data_obj),
        to_dict=lambda: {
            "id": event_id,
            "type": event_type,
            "data": {"object": data_obj},
            "created": created,
        },
    )


def test_config_updates_written(tmp_path, monkeypatch):
    os.environ.setdefault("MENACE_ID", "test")
    resolve_path("billing")
    os.environ.setdefault("STRIPE_API_KEY", "sk")

    config_file = tmp_path / "cfg.json"

    monkeypatch.setattr(msl, "_BILLING_EVENT_DB", None)
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: None)

    event = _make_event("evt1", "charge.refunded", "bot-rogue", created=1)
    monkeypatch.setattr(rad, "_iter_recent_events", lambda hours: [event])
    monkeypatch.setattr(rad, "load_whitelist", lambda path: set())
    monkeypatch.setattr(rad.billing_logger, "log_event", lambda **kw: None)

    class DummyDB:
        def __init__(self, *a, **k):
            self.conn = SimpleNamespace()

    monkeypatch.setattr(rad, "BillingLogDB", lambda *a, **k: DummyDB())

    def record(event_type, metadata, instruction, **kwargs):
        metadata = {**metadata, "config_updates": {"block_unlogged_charges": True}}
        return msl.record_billing_event(event_type, metadata, instruction, **kwargs)

    monkeypatch.setattr(rad, "record_billing_event", record)
    monkeypatch.setattr(rad, "record_payment_anomaly", lambda *a, **k: None)

    updates: list = []
    engine = SimpleNamespace(
        update_generation_params=lambda meta: updates.append(meta)
    )

    rad.detect_anomalies(
        self_coding_engine=engine,
        config_path=config_file,
    )

    assert json.loads(config_file.read_text()) == {"block_unlogged_charges": True}
    assert updates and updates[0]["config_updates"]["block_unlogged_charges"] is True

