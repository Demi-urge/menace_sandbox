import sandbox_runner.cli as cli

def test_adaptive_threshold():
    values = [1.0, 2.0, 3.0]
    thr = cli._adaptive_threshold(values, 3, factor=1.0)
    assert thr > 0


def test_adaptive_synergy_threshold():
    hist = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    thr = cli._adaptive_synergy_threshold(hist, 3, factor=1.0)
    assert thr > 0
