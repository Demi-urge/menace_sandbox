import sandbox_runner.cli as cli


def test_adaptive_synergy_convergence_converges():
    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.002},
        {"synergy_roi": 0.0015},
        {"synergy_roi": 0.0011},
        {"synergy_roi": 0.0009},
    ]

    ok, ema, conf = cli.adaptive_synergy_convergence(hist, 5, threshold=0.01)
    assert ok is True
    assert conf >= 0.95


def test_adaptive_synergy_convergence_not_converged():
    hist = [
        {"synergy_roi": 0.05},
        {"synergy_roi": -0.05},
        {"synergy_roi": 0.05},
        {"synergy_roi": -0.05},
    ]

    ok, _, conf = cli.adaptive_synergy_convergence(hist, 4, threshold=0.01)
    assert ok is False
    assert conf < 0.95

