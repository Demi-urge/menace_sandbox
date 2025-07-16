import pytest
import menace.prediction_manager_bot as pmb
import menace.future_prediction_bots as fpb
import menace.data_bot as db
import menace.roi_tracker as rt


def test_manager_registers_future_bots(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(metrics_db)
    manager = pmb.PredictionManager(tmp_path / "reg.json", data_bot=data_bot)
    assert any(
        isinstance(e.bot, fpb.FutureLucrativityBot) for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureProfitabilityBot) for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureAntifragilityBot) for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureShannonEntropyBot)
        for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureSynergyProfitBot) for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureSynergyMaintainabilityBot)
        for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureSynergyCodeQualityBot)
        for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureSynergyNetworkLatencyBot)
        for e in manager.registry.values()
    )
    assert any(
        isinstance(e.bot, fpb.FutureSynergyThroughputBot)
        for e in manager.registry.values()
    )


def test_roi_tracker_predicts_with_future_bots(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m2.db")
    data_bot = db.DataBot(metrics_db)
    manager = pmb.PredictionManager(tmp_path / "reg2.json", data_bot=data_bot)
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "projected_lucrativity": 0.5,
                "profitability": 0.6,
                "antifragility": 0.7,
                "shannon_entropy": 0.8,
                "flexibility": 0.9,
                "synergy_profitability": 0.2,
                "synergy_revenue": 0.1,
                "synergy_projected_lucrativity": 0.3,
                "synergy_maintainability": 0.4,
                "synergy_code_quality": 0.5,
                "synergy_network_latency": 1.2,
                "synergy_throughput": 2.3,
            }
        ]
    )
    data_bot.db.fetch = lambda limit=20: df  # type: ignore
    tracker = rt.ROITracker()
    tracker.update(
        0.0,
        0.0,
        metrics={
            "projected_lucrativity": 0.5,
            "profitability": 0.6,
            "antifragility": 0.7,
            "shannon_entropy": 0.8,
            "flexibility": 0.9,
            "synergy_profitability": 0.2,
            "synergy_revenue": 0.1,
            "synergy_projected_lucrativity": 0.3,
            "synergy_maintainability": 0.4,
            "synergy_code_quality": 0.5,
            "synergy_network_latency": 1.2,
            "synergy_throughput": 2.3,
        },
    )
    tracker.predict_all_metrics(manager)
    assert "projected_lucrativity" in tracker.predicted_metrics
    assert "profitability" in tracker.predicted_metrics
    assert "antifragility" in tracker.predicted_metrics
    assert "shannon_entropy" in tracker.predicted_metrics
    assert "flexibility" in tracker.predicted_metrics
    assert "synergy_profitability" in tracker.predicted_metrics
    assert "synergy_revenue" in tracker.predicted_metrics
    assert "synergy_projected_lucrativity" in tracker.predicted_metrics
    assert "synergy_maintainability" in tracker.predicted_metrics
    assert "synergy_code_quality" in tracker.predicted_metrics
    assert "synergy_network_latency" in tracker.predicted_metrics
    assert "synergy_throughput" in tracker.predicted_metrics
    assert tracker.rolling_mae_metric("flexibility") >= 0.0
    assert tracker.rolling_mae_metric("antifragility") >= 0.0
    assert tracker.rolling_mae_metric("shannon_entropy") >= 0.0
    assert tracker.rolling_mae_metric("synergy_maintainability") >= 0.0
    assert tracker.rolling_mae_metric("synergy_code_quality") >= 0.0
    assert tracker.rolling_mae_metric("synergy_network_latency") >= 0.0
    assert tracker.rolling_mae_metric("synergy_throughput") >= 0.0
