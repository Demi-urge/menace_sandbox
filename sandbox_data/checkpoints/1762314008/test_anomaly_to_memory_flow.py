from pathlib import Path
from db_router import init_db_router


def _setup(monkeypatch, tmp_path):
    """Configure temporary databases and stub out side effects."""

    # Provide predictable paths for modules that rely on dynamic_path_router
    import dynamic_path_router

    def fake_resolve(name):
        return tmp_path / str(name)

    def fake_resolve_dir(name):
        path = tmp_path / str(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Provide stub for unified_event_bus expected by menace_sanity_layer
    (tmp_path / "unified_event_bus.py").write_text(
        "class UnifiedEventBus:\n    def publish(self, *a, **k):\n        pass\n"
    )

    monkeypatch.setattr(dynamic_path_router, "resolve_path", fake_resolve)
    monkeypatch.setattr(dynamic_path_router, "resolve_dir", fake_resolve_dir)
    monkeypatch.setattr(
        dynamic_path_router, "get_project_root", lambda: tmp_path, raising=False
    )
    monkeypatch.setattr(
        dynamic_path_router, "get_project_roots", lambda: [tmp_path], raising=False
    )

    # Stub discrepancy_db to avoid heavy imports
    import types, sys

    discrepancy_stub = types.ModuleType("discrepancy_db")

    class DummyDiscrepancyDB:
        def __init__(self, *a, **k):
            pass

        def log_detection(self, *a, **k):
            pass

    discrepancy_stub.DiscrepancyDB = DummyDiscrepancyDB
    discrepancy_stub.DiscrepancyRecord = None
    sys.modules.setdefault("discrepancy_db", discrepancy_stub)

    # Configure temporary databases for anomaly logging
    init_db_router("ba", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    import menace_sanity_layer as msl

    # Avoid external side effects
    msl._DISCREPANCY_DB = None
    msl.GPT_MEMORY_MANAGER = None
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)
    msl._EVENT_BUS = msl.UnifiedEventBus()

    class DummyMM:
        def __init__(self) -> None:
            self.stored: list[tuple[str, dict, str]] = []

        def query(self, key, limit):  # pragma: no cover - unused
            return []

        def store(self, key, data, tags=""):
            self.stored.append((key, data, tags))

    mm = DummyMM()
    msl._MEMORY_MANAGER = mm

    import stripe_watchdog as sw

    # Prevent file I/O during anomaly emission
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)

    return mm, sw, msl


def test_detect_missing_charge_records_memory(monkeypatch, tmp_path):
    """End-to-end anomaly detection to GPT memory logging."""

    mm, sw, msl = _setup(monkeypatch, tmp_path)

    charges = [
        {"id": "ch_1", "amount": 5, "receipt_email": "a@example.com", "created": 1}
    ]

    # No ledger entries so the charge is considered missing
    anomalies = sw.detect_missing_charges(
        charges, [], write_codex=False, export_training=False
    )

    assert anomalies and anomalies[0]["type"] == "missing_charge"
    assert mm.stored
    key, data, tags = mm.stored[0]
    assert key == "billing:missing_charge"
    assert tags == "billing,anomaly"
    assert (
        data["instruction"]
        == msl.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    )


def test_detect_missing_refund_records_memory(monkeypatch, tmp_path):
    """Missing refunds trigger memory logging with correct tags."""

    mm, sw, msl = _setup(monkeypatch, tmp_path)

    refunds = [{"id": "re_1", "amount": 3, "charge": "ch_1", "account": "acct"}]

    anomalies = sw.detect_missing_refunds(
        refunds, [], [], write_codex=False, export_training=False
    )

    assert anomalies and anomalies[0]["type"] == "missing_refund"
    assert mm.stored
    key, data, tags = mm.stored[0]
    assert key == "billing:missing_refund"
    assert tags == "billing,anomaly"
    assert (
        data["instruction"]
        == msl.EVENT_TYPE_INSTRUCTIONS["missing_refund"]
    )


def test_disabled_webhook_memory(monkeypatch, tmp_path):
    """Disabled webhook endpoints are logged to memory."""

    mm, sw, msl = _setup(monkeypatch, tmp_path)

    monkeypatch.setattr(sw.alert_dispatcher, "dispatch_alert", lambda *a, **k: None)

    class DummyStripe:
        class WebhookEndpoint:
            @staticmethod
            def list(api_key):
                return [
                    {
                        "id": "wh_1",
                        "url": "https://example.com/hook",
                        "status": "disabled",
                        "account": "acct",
                    }
                ]

    monkeypatch.setattr(sw, "stripe", DummyStripe)

    flagged = sw.check_webhook_endpoints("sk_test", approved=["wh_1"])

    assert flagged == ["wh_1"]
    assert mm.stored
    key, data, tags = mm.stored[0]
    assert key == "billing:disabled_webhook"
    assert tags == "billing,anomaly"
    assert (
        data["instruction"]
        == msl.EVENT_TYPE_INSTRUCTIONS["disabled_webhook"]
    )
