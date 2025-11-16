import json
import os
import sys
import types
from types import SimpleNamespace

from dynamic_path_router import resolve_path


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

    sys.modules["menace_sanity_layer"] = msl = types.SimpleNamespace(
        _BILLING_EVENT_DB=None,
        _get_gpt_memory=lambda: None,
        _get_memory_manager=lambda: None,
        ANOMALY_THRESHOLDS={},
        ANOMALY_HINTS={},
        record_billing_event=lambda *a, **k: None,
    )
    from billing import refund_anomaly_detector as rad
    monkeypatch.setattr(msl, "_BILLING_EVENT_DB", None)
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: None)

    mm_storage: dict[str, dict] = {}

    class DummyMM:
        def query(self, key, limit):
            if key in mm_storage:
                return [SimpleNamespace(data=json.dumps(mm_storage[key]))]
            return []

        def store(self, key, data, tags=""):
            mm_storage[key] = data

    mm = DummyMM()
    monkeypatch.setattr(msl, "_get_memory_manager", lambda: mm)
    monkeypatch.setitem(msl.ANOMALY_THRESHOLDS, "refund_anomaly", 1)
    monkeypatch.setitem(msl.ANOMALY_HINTS, "refund_anomaly", {"custom_hint": True})

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
        config_file.write_text(json.dumps({"block_unlogged_charges": True}))
        engine.update_generation_params({"custom_hint": True, "event_type": event_type})
        return metadata

    monkeypatch.setattr(rad, "record_billing_event", record)
    monkeypatch.setattr(rad, "record_payment_anomaly", lambda *a, **k: None)

    updates: list = []
    engine = SimpleNamespace(
        update_generation_params=lambda meta: updates.append(meta)
    )

    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    rad.detect_anomalies(
        self_coding_engine=engine,
        config_path=config_file,
        context_builder=builder,
    )

    assert json.loads(config_file.read_text()) == {"block_unlogged_charges": True}
    assert updates == [{"custom_hint": True, "event_type": "refund_anomaly"}]
