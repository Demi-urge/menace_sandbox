import pytest

torch = pytest.importorskip("torch")
learners = pytest.importorskip("menace.self_improvement.learners")


class DummyLearner(learners._BaseRLSynergyLearner):
    def update(self, roi_delta, deltas, extra=None):
        self.weights["roi"] += roi_delta


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        learners._BaseRLSynergyLearner()  # type: ignore[abstract]


def test_dummy_learner_updates(tmp_path):
    learner = DummyLearner(path=tmp_path / "w.json")
    start = learner.weights["roi"]
    learner.update(1.0, {"synergy_roi": 1.0})
    assert learner.weights["roi"] > start


def _run_learner(cls, tmp_path):
    learner = cls(path=tmp_path / "w.json", hidden_sizes=[8], batch_size=1)
    deltas = {name: 0.0 for name in learner.names}
    learner.update(0.1, deltas)
    assert all(0.0 <= learner.weights[k] <= 10.0 for k in learner.weights)


def test_sac_learner_runs(tmp_path):
    _run_learner(learners.SACSynergyLearner, tmp_path)


def test_td3_learner_runs(tmp_path):
    _run_learner(learners.TD3SynergyLearner, tmp_path)
