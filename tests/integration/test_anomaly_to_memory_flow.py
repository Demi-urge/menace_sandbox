from db_router import init_db_router


def test_detect_missing_charge_records_memory(monkeypatch, tmp_path):
    """End-to-end anomaly detection to GPT memory logging."""

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

    charges = [
        {"id": "ch_1", "amount": 5, "receipt_email": "a@example.com", "created": 1}
    ]

    # No ledger entries so the charge is considered missing
    anomalies = sw.detect_missing_charges(
        charges, [], write_codex=False, export_training=False
    )

    assert anomalies and anomalies[0]["type"] == "missing_charge"
    assert mm.stored and mm.stored[0][0] == "billing:missing_charge"
