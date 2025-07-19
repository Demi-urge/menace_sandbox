import sandbox_runner.cli as cli

def test_synergy_converged():
    hist = [
        {"synergy_roi": 0.005},
        {"synergy_roi": 0.005},
        {"synergy_roi": 0.005},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 3, 0.03)
    assert ok is True
    assert conf >= 0.95

def test_synergy_not_converged():
    hist = [
        {"synergy_roi": 0.05},
        {"synergy_roi": -0.04},
        {"synergy_roi": 0.04},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 3, 0.03)
    assert ok is False
    assert conf < 0.95
