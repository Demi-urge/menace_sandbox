import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.policy_learning import PolicyLearner


def test_policy_learning_updates_bandit_and_snapshots():
    actions = ["a", "b"]
    tactics = {"a": "prov", "b": "flat"}
    learner = PolicyLearner(actions, tactics, state_dim=2, lr=0.5)
    act, style, conf = learner.act([1.0, 0.0])
    assert act in actions
    learner.learn(reward=1.0, ctr=0.3, sentiment=0.2, session=0.1)
    bandit = learner.scouts[tactics[act]]
    assert bandit.weights.get(act) is not None
    assert len(learner.brain.snapshots) == 1
    act2, style2, conf2 = learner.act([1.0, 0.0])
    assert act2 in actions
    assert style2 in {"punch", "hedge", "probe"}


def test_train_from_dataset_updates_params(tmp_path):
    data = [
        {"state": [1, 0], "action": "a", "reward": 1.0},
        {"state": [0, 1], "action": "b", "reward": 0.5},
    ]
    path = tmp_path / "data.json"
    import json

    path.write_text(json.dumps(data))

    actions = ["a", "b"]
    tactics = {"a": "prov", "b": "flat"}
    learner = PolicyLearner(actions, tactics, state_dim=2, lr=0.5)
    before = [row[:] for row in learner.brain.params]
    learner.train_from_dataset(str(path))
    assert learner.brain.params != before
