import menace.energy_forecast_bot as efb
import menace.prediction_manager_bot as pmb
import menace.data_bot as db
import menace.capital_management_bot as cmb


def test_forecast_and_integration(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(metrics_db)
    manager = pmb.PredictionManager(tmp_path / "reg.json", data_bot=data_bot)
    capital = cmb.CapitalManagementBot(data_bot=data_bot, prediction_manager=manager)
    manager.capital_bot = capital
    forecast = efb.EnergyForecastBot(data_bot=data_bot, capital_bot=capital)
    bid = manager.register_bot(forecast, {"scope": ["energy"]})
    capital.assigned_prediction_bots.append(bid)

    val = forecast.predict([0.5, 0.1, 0.2, 0.8, 0.7, 0.1])
    assert 0.0 <= val <= 1.0

    energy = capital.energy_score(
        load=0.2, success_rate=0.9, deploy_eff=0.8, failure_rate=0.1
    )
    assert 0.0 <= energy <= 1.0
