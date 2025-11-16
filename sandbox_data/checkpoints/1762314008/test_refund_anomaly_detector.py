from __future__ import annotations

import json
import os
from types import SimpleNamespace

from dynamic_path_router import resolve_path


def _make_event(
    event_id: str, event_type: str, bot_id: str, *, created: int
) -> SimpleNamespace:
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


def test_detects_unlogged_and_unauthorized(tmp_path, monkeypatch):
    os.environ.setdefault("MENACE_ID", "test")
    resolve_path("billing")

    import sys
    import types

    payment_calls: list[tuple[tuple, dict]] = []
    billing_calls: list[tuple[tuple, dict]] = []

    fake_layer = types.SimpleNamespace(
        record_payment_anomaly=lambda *a, **k: payment_calls.append((a, k)),
        record_billing_event=lambda *a, **k: billing_calls.append((a, k)),
    )
    sys.modules["menace_sanity_layer"] = fake_layer

    from billing.billing_log_db import BillingLogDB, BillingEvent
    from billing import refund_anomaly_detector as rad

    whitelist = tmp_path / "whitelist.json"
    whitelist.write_text(json.dumps(["bot-approved"]))

    db_path = tmp_path / "billing.db"
    db = BillingLogDB(db_path)
    db.log(
        BillingEvent(
            action="refund",
            bot_id="bot-approved",
            ts="1970-01-01 00:00:01",
        )
    )

    events = [
        _make_event("evt_logged", "charge.refunded", "bot-approved", created=1),
        _make_event("evt_unlogged", "charge.refunded", "bot-approved", created=3600),
        _make_event(
            "evt_unauth",
            "payment_intent.payment_failed",
            "bot-rogue",
            created=7200,
        ),
    ]

    monkeypatch.setattr(rad, "_iter_recent_events", lambda hours: events)

    logged: list[dict] = []
    monkeypatch.setattr(rad.billing_logger, "log_event", lambda **kw: logged.append(kw))

    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    anomalies = rad.detect_anomalies(
        hours=1, whitelist_path=whitelist, db_path=db_path, context_builder=builder
    )

    assert {a["reason"] for a in anomalies} == {"unlogged", "unauthorized"}
    assert {a["id"] for a in anomalies} == {"evt_unlogged", "evt_unauth"}
    assert {e["id"] for e in logged} == {"evt_unlogged", "evt_unauth"}

    assert {a[0][0] for a in payment_calls} == {"missing_refund", "unauthorized_failure"}
    assert {
        a[0][1]["stripe_event_id"]
        for a in payment_calls
        if a[0][0] == "missing_refund"
    } == {"evt_unlogged"}
    assert {
        a[0][1]["stripe_event_id"]
        for a in payment_calls
        if a[0][0] == "unauthorized_failure"
    } == {"evt_unauth"}

    assert {a[0][0] for a in billing_calls} == {"refund_anomaly", "payment_failure"}
    assert {
        a[0][1]["stripe_event_id"] for a in billing_calls if a[0][0] == "refund_anomaly"
    } == {"evt_unlogged"}
    assert {
        a[0][1]["stripe_event_id"] for a in billing_calls if a[0][0] == "payment_failure"
    } == {"evt_unauth"}
    assert {a[0][1]["reason"] for a in billing_calls if a[0][0] == "refund_anomaly"} == {
        "unlogged"
    }
    assert {a[0][1]["reason"] for a in billing_calls if a[0][0] == "payment_failure"} == {
        "unauthorized"
    }
