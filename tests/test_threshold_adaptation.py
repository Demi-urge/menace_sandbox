import types

import menace.data_bot as db


def test_thresholds_persist_with_forecasts(tmp_path, monkeypatch):
    metrics_db = db.MetricsDB(path=tmp_path / "metrics.db")
    bot = db.DataBot(db=metrics_db, threshold_update_interval=0)

    saved: list[tuple[float | None, float | None, float | None]] = []

    def fake_update(bot_name, *, roi_drop=None, error_increase=None, test_failure_increase=None, **_):
        saved.append((roi_drop, error_increase, test_failure_increase))

    monkeypatch.setattr(db, "persist_sc_thresholds", fake_update)

    class FakePredictor:
        def __init__(self) -> None:
            self.roi_seq = iter([1.0, 0.9, 0.8])
            self.err_seq = iter([0.0, 0.2, 0.4])

        def train(self) -> None:  # pragma: no cover - simple stub
            pass

        def predict_future_metrics(self):  # pragma: no cover - simple stub
            return types.SimpleNamespace(roi=next(self.roi_seq), errors=next(self.err_seq))

    bot.trend_predictor = FakePredictor()

    for _ in range(3):
        bot.check_degradation("alpha", roi=1.0, errors=0.0)

    assert saved, "update_thresholds should be invoked"
    th = bot.get_thresholds("alpha")
    assert (th.roi_drop, th.error_threshold, th.test_failure_threshold) == saved[-1]

