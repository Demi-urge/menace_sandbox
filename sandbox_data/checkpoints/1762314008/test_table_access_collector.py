from menace import telemetry_backend as tb


def test_table_access_collector_flush() -> None:
    tb.record_table_access("alpha", "bots", "shared", 2)
    tb.record_table_access("alpha", "bots", "shared", 1)
    tb.record_table_access("beta", "memory", "local", 5)

    counts = tb.get_table_access_counts()
    assert counts["alpha"]["shared"]["bots"] == 3
    assert counts["beta"]["local"]["memory"] == 5

    snapshot = tb.get_table_access_counts(flush=True)
    assert snapshot == counts
    assert tb.get_table_access_counts() == {}
