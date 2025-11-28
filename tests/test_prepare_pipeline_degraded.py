import coding_bot_interface as cbi


def test_deferred_pending_gates_filtered_from_critical_pool():
    ready, pending, deferred = cbi._filter_deferred_pending_gates(
        ["pipeline_config", "background_loops"],
        ["pipeline_config", "background_loops", "db_indexes"],
        ["background_loops", "db_indexes"],
    )

    assert "background_loops" not in ready
    assert "background_loops" not in pending
    assert "db_indexes" not in pending
    assert deferred == {"background_loops", "db_indexes"}

    ready2, pending2, deferred2 = cbi._filter_deferred_pending_gates(
        [],
        ["pipeline_config"],
        ["pipeline_config"],
    )
    assert "pipeline_config" in pending2
    assert deferred2 == {"pipeline_config"}
