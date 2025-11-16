import types
import sys

sys.modules.setdefault(
    "menace.run_autonomous",
    types.SimpleNamespace(
        _verify_required_dependencies=lambda: None, LOCAL_KNOWLEDGE_MODULE=None
    ),
)

import sandbox_runner.cli as cli
import pytest

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


def test_diminishing_modules_expected_confidence():
    history = {"mod": [0.002, 0.002, 0.002]}
    flags, conf = cli._diminishing_modules(history, set(), 0.01, consecutive=3)
    assert flags == ["mod"]
    assert conf["mod"] == pytest.approx(1.0)


def test_diminishing_modules_above_threshold():
    history = {"mod": [0.02, 0.02, 0.02]}
    flags, conf = cli._diminishing_modules(history, set(), 0.01, consecutive=3)
    assert flags == []
    assert conf == {}


def test_diminishing_modules_entropy():
    history = {}
    entropy_history = {"mod": [0.005, 0.004, 0.003]}
    flags, conf = cli._diminishing_modules(
        history, set(), 0.01, consecutive=3, entropy_history=entropy_history
    )
    assert flags == ["mod"]
    assert conf["mod"] == pytest.approx(1.0)


def test_diminishing_modules_entropy_custom_params():
    history = {}
    entropy_history = {"mod": [0.02, 0.004, 0.003]}
    flags, _ = cli._diminishing_modules(
        history, set(), 0.01, consecutive=3, entropy_history=entropy_history
    )
    assert flags == []
    flags, _ = cli._diminishing_modules(
        history,
        set(),
        0.01,
        consecutive=3,
        entropy_history=entropy_history,
        entropy_threshold=0.005,
        entropy_consecutive=2,
    )
    assert flags == ["mod"]
