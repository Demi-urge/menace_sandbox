from .test_stripe_billing_router_logging import _import_module


def test_alert_mismatch_logs_error_and_rolls_back(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    rollback_calls = []

    class DummyRM:
        def rollback(self, tag, requesting_bot=None):
            rollback_calls.append((tag, requesting_bot))

    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: DummyRM())

    log_call = {}

    def fake_log(bot_id, message):
        log_call["args"] = (bot_id, message)
        sbr.rollback_manager.RollbackManager().rollback("latest", requesting_bot=bot_id)

    monkeypatch.setattr(sbr, "log_critical_discrepancy", fake_log)

    log_event_data = {}
    monkeypatch.setattr(
        sbr.billing_logger, "log_event", lambda **kw: log_event_data.update(kw)
    )

    billing_event_data = {}
    monkeypatch.setattr(
        sbr, "log_billing_event", lambda action, **kw: billing_event_data.update({"action": action, **kw})
    )

    sbr._alert_mismatch("bot123", "acct_bad", amount=7.5)

    assert rollback_calls == [("latest", "bot123")]
    assert log_call["args"] == ("bot123", "Stripe account mismatch")
    assert log_event_data["error"] is True
    assert log_event_data["bot_id"] == "bot123"
    assert log_event_data["destination_account"] == "acct_bad"
    assert log_event_data["amount"] == 7.5
    assert billing_event_data == {
        "action": "mismatch",
        "bot_id": "bot123",
        "amount": 7.5,
        "destination_account": "acct_bad",
    }
