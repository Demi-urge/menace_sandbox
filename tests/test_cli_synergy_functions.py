import sandbox_runner.cli as cli


def test_synergy_threshold_variance_decrease():
    high_var_hist = [
        {"synergy_roi": 0.5},
        {"synergy_roi": -0.5},
        {"synergy_roi": 0.5},
        {"synergy_roi": -0.5},
    ]
    high_var_preds = [{"synergy_roi": 0.0}] * 4
    high_thr = cli._adaptive_synergy_threshold(
        high_var_hist, 4, factor=1.0, predictions=high_var_preds
    )

    low_var_hist = [
        {"synergy_roi": 0.1},
        {"synergy_roi": -0.1},
        {"synergy_roi": 0.1},
        {"synergy_roi": -0.1},
    ]
    low_var_preds = [{"synergy_roi": 0.0}] * 4
    low_thr = cli._adaptive_synergy_threshold(
        low_var_hist, 4, factor=1.0, predictions=low_var_preds
    )

    assert low_thr < high_thr


def test_synergy_converged_stable_series():
    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 5, 0.01)
    assert ok is True
    assert conf >= 0.95
