import pickle
import pytest

import menace.self_improvement as sie


def test_dqn_weight_persistence_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(sie, "sip_torch", None)
    path = tmp_path / "w.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.5)
    start = learner.weights.copy()
    deltas = {
        "synergy_roi": 0.5,
        "synergy_efficiency": 0.2,
        "synergy_resilience": -0.3,
        "synergy_antifragility": 0.1,
    }
    learner.update(1.0, deltas)
    changed = learner.weights.copy()
    assert changed != start
    learner2 = sie.DQNSynergyLearner(path=path, lr=0.5)
    assert learner2.weights == pytest.approx(changed)


def test_policy_pickle_called(tmp_path, monkeypatch):
    monkeypatch.setattr(sie, "sip_torch", None)
    path = tmp_path / "w.json"
    called = {"dump": 0, "load": 0}
    real_dump = pickle.dump
    real_load = pickle.load

    def wrap_dump(obj, fh, *a, **k):
        called["dump"] += 1
        return real_dump(obj, fh, *a, **k)

    def wrap_load(fh, *a, **k):
        called["load"] += 1
        return real_load(fh, *a, **k)

    monkeypatch.setattr(sie.pickle, "dump", wrap_dump)
    monkeypatch.setattr(sie.pickle, "load", wrap_load)

    learner = sie.DQNSynergyLearner(path=path, lr=0.1)
    learner.update(0.5, {"synergy_roi": 1.0})
    assert called["dump"] >= 1

    sie.DQNSynergyLearner(path=path, lr=0.1)
    assert called["load"] >= 1
