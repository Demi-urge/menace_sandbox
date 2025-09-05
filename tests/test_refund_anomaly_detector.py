from __future__ import annotations

import json
from types import SimpleNamespace

from billing.billing_log_db import BillingLogDB, BillingEvent
from billing import refund_anomaly_detector as rad
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
    resolve_path("billing")
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
    recorded: list[tuple[str, dict, str]] = []
    monkeypatch.setattr(
        rad.menace_sanity_layer,
        "record_payment_anomaly",
        lambda et, md, instr, **kw: recorded.append((et, md, instr)),
    )

    anomalies = rad.detect_anomalies(hours=1, whitelist_path=whitelist, db_path=db_path)

    assert {a["reason"] for a in anomalies} == {"unlogged", "unauthorized"}
    assert {a["id"] for a in anomalies} == {"evt_unlogged", "evt_unauth"}
    assert {e["id"] for e in logged} == {"evt_unlogged", "evt_unauth"}
    assert {r[0] for r in recorded} == {"unlogged", "unauthorized"}
    assert {
        r[1]["stripe_event_id"] for r in recorded if r[0] == "unlogged"
    } == {"evt_unlogged"}
    assert {
        r[1]["stripe_event_id"] for r in recorded if r[0] == "unauthorized"
    } == {"evt_unauth"}
