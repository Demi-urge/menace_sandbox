import sandbox_runner.cli as cli

def test_diminishing_modules_stable():
    history = {"mod": [0.005, 0.005, 0.005, 0.005]}
    flags, conf = cli._diminishing_modules(history, set(), 0.01, consecutive=3)
    assert flags == ["mod"]
    assert conf["mod"] >= 0.95


def test_diminishing_modules_noisy():
    history = {"mod": [0.05, -0.05, 0.05, -0.05, 0.05]}
    flags, conf = cli._diminishing_modules(history, set(), 0.01, consecutive=3)
    assert flags == []
    assert conf == {}
