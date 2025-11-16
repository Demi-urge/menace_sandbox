from .test_stripe_billing_router_logging import _import_module


def _stub_unified_event_bus(monkeypatch, tmp_path):
    import dynamic_path_router
    from dynamic_path_router import resolve_path as _orig_resolve

    stub_path = tmp_path / "unified_event_bus.py"  # path-ignore
    stub_path.write_text("class UnifiedEventBus:\n    pass\n")

    def fake_resolve(name, root=None):
        if name == "unified_event_bus.py":  # path-ignore
            return stub_path
        try:
            return _orig_resolve(name, root)
        except TypeError:
            return _orig_resolve(name)

    monkeypatch.setattr(dynamic_path_router, "resolve_path", fake_resolve)


def test_alert_mismatch_logs_error_and_rolls_back(monkeypatch, tmp_path):
    _stub_unified_event_bus(monkeypatch, tmp_path)
    sbr = _import_module(monkeypatch, tmp_path)
    # Prevent any environment-derived account identifiers from influencing the
    # module under test and force ``_get_account_id`` to return the hardcoded
    # master account.
    monkeypatch.setattr(
        sbr, "_get_account_id", lambda api_key: sbr.STRIPE_MASTER_ACCOUNT_ID
    )

    sbr.sandbox_review.reset()
    rollback_calls = []

    class DummyRM:
        def rollback(self, tag, requesting_bot=None):
            rollback_calls.append((tag, requesting_bot))

    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: DummyRM())

    log_calls = []

    def fake_log(*args):
        log_calls.append(args)
        assert args == ("bot123", "Stripe account mismatch")
        sbr.rollback_manager.RollbackManager().rollback("latest", requesting_bot=args[0])

    monkeypatch.setattr(sbr, "log_critical_discrepancy", fake_log)

    log_event_data = {}
    monkeypatch.setattr(
        sbr.billing_logger, "log_event", lambda **kw: log_event_data.update(kw)
    )

    billing_event_data = {}
    monkeypatch.setattr(
        sbr,
        "log_billing_event",
        lambda action, **kw: billing_event_data.update({"action": action, **kw}),
    )
    import evolution_lock_flag

    lock_calls = []
    monkeypatch.setattr(
        evolution_lock_flag,
        "trigger_lock",
        lambda reason, severity: lock_calls.append((reason, severity)),
    )

    recorded: list[tuple[str, dict, str]] = []
    monkeypatch.setattr(
        sbr,
        "record_payment_anomaly",
        lambda et, md, instr, **kw: recorded.append((et, md, instr)),
    )
    billing_events: list[tuple[str, dict, str]] = []
    monkeypatch.setattr(
        sbr,
        "record_billing_event",
        lambda et, md, instr, **kw: billing_events.append((et, md, instr)),
    )
    sanity_events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        sbr.menace_sanity_layer,
        "record_event",
        lambda et, md: sanity_events.append((et, md)),
    )

    sbr._alert_mismatch("bot123", "acct_bad", amount=7.5)

    assert rollback_calls == [("latest", "bot123")]
    assert log_calls == [("bot123", "Stripe account mismatch")]
    assert lock_calls == [("Stripe account mismatch for bot123", 5)]
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
    assert recorded and recorded[0][0] == "stripe_account_mismatch"
    assert recorded[0][1]["bot_id"] == "bot123"
    assert recorded[0][1]["account_id"] == "acct_bad"
    assert billing_events and billing_events[0][0] == "stripe_account_mismatch"
    assert billing_events[0][1]["bot_id"] == "bot123"
    assert billing_events[0][1]["account_id"] == "acct_bad"
    assert sanity_events == [
        (
            "account_mismatch",
            {"bot_id": "bot123", "destination_account": "acct_bad", "amount": 7.5},
        )
    ]
    assert sbr.sandbox_review.is_paused("bot123")
