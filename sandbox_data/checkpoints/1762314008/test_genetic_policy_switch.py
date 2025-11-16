import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.policy_learning import PolicyLearner


def test_policy_learner_switches_modes():
    actions = ["x"]
    tactics = {"x": "t"}
    learner = PolicyLearner(
        actions,
        tactics,
        state_dim=1,
        lr=0.1,
        min_epochs=1,
        plateau_patience=1,
        ga_generations=1,
        ga_population=2,
    )
    learner.act([1.0])
    learner.learn(reward=0.0)
    assert learner.mode == "ga"
    learner.learn(reward=0.0)
    assert learner.mode == "rl"
