import menace.data_bot as db
from menace.roi_thresholds import ROIThresholds


def test_thresholds_adjust_upward_after_improvements(tmp_path, monkeypatch):
    metrics_db = db.MetricsDB(path=tmp_path / "metrics.db")
    bot = db.DataBot(db=metrics_db, threshold_update_interval=0)

    outputs = iter([
        ROIThresholds(roi_drop=-0.2, error_threshold=1.0, test_failure_threshold=0.0),
        ROIThresholds(roi_drop=-0.1, error_threshold=0.5, test_failure_threshold=0.0),
        ROIThresholds(roi_drop=-0.05, error_threshold=0.25, test_failure_threshold=0.0),
    ])

    saved = []

    def fake_adaptive(*args, **kwargs):
        return next(outputs)

    def fake_save(bot_name, *, roi_drop=None, error_increase=None, test_failure_increase=None, **_):
        saved.append((roi_drop, error_increase, test_failure_increase))

    monkeypatch.setattr(db, "adaptive_thresholds", fake_adaptive)
    monkeypatch.setattr(db, "save_sc_thresholds", fake_save)

    for _ in range(3):
        bot.check_degradation("alpha", roi=1.0, errors=0.0)

    th = bot.get_thresholds("alpha")
    assert th.roi_drop == -0.05
    assert th.error_threshold == 0.25
    assert saved[-1] == (-0.05, 0.25, 0.0)


def test_thresholds_adjust_downward_after_regressions(tmp_path, monkeypatch):
    metrics_db = db.MetricsDB(path=tmp_path / "metrics.db")
    bot = db.DataBot(db=metrics_db, threshold_update_interval=0)

    outputs = iter([
        ROIThresholds(roi_drop=-0.05, error_threshold=0.5, test_failure_threshold=0.0),
        ROIThresholds(roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0),
        ROIThresholds(roi_drop=-0.2, error_threshold=2.0, test_failure_threshold=0.0),
    ])

    saved = []

    def fake_adaptive(*args, **kwargs):
        return next(outputs)

    def fake_save(bot_name, *, roi_drop=None, error_increase=None, test_failure_increase=None, **_):
        saved.append((roi_drop, error_increase, test_failure_increase))

    monkeypatch.setattr(db, "adaptive_thresholds", fake_adaptive)
    monkeypatch.setattr(db, "save_sc_thresholds", fake_save)

    for _ in range(3):
        bot.check_degradation("beta", roi=1.0, errors=0.0)

    th = bot.get_thresholds("beta")
    assert th.roi_drop == -0.2
    assert th.error_threshold == 2.0
    assert saved[-1] == (-0.2, 2.0, 0.0)
