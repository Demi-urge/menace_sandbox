import menace.revenue_amplifier as ra


def test_spike_detection_and_rebalance(tmp_path):
    rev_db = ra.RevenueEventsDB(tmp_path / "rev.db")
    profit_db = ra.ProfitabilityDB(tmp_path / "prof.db")
    spike_monitor = ra.SalesSpikeMonitor(rev_db)
    eval_bot = ra.RevenueSpikeEvaluatorBot(rev_db, window=5, threshold=2.0)
    alloc_bot = ra.CapitalAllocationBot(profit_db)

    # baseline sales
    for _ in range(4):
        spike_monitor.record_sale("model1", 10.0, "platform", "seg")
    # spike
    spike_monitor.record_sale("model1", 50.0, "platform", "seg")

    assert eval_bot.detect_spike("model1")
    alloc_bot.rebalance("model1", 50.0)
    rows = profit_db.conn.execute("SELECT COUNT(*) FROM profit").fetchone()
    assert rows[0] == 1
