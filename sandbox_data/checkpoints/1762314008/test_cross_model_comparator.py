from menace.cross_model_comparator import CrossModelComparator
from menace.model_deployer import ModelDeployer


class DummyPathways:
    def __init__(self, scores=None):
        self.scores = scores or {}

    def similar_actions(self, action: str, limit: int = 1):
        name = action.split("run_cycle:", 1)[1]
        if name in self.scores:
            return [(1, self.scores[name])]
        return []

    def highest_myelination_score(self):
        return max(self.scores.values()) if self.scores else 0.0


class DummyHistory:
    def __init__(self, weights=None):
        self.weights = weights or {"A": 1.0, "B": 0.5}

    def deployment_weights(self):
        return self.weights


class DummyDeployer(ModelDeployer):
    def __init__(self):
        self.deployed = []
        self.cloned = []

    def deploy_model(self, name: str):
        self.deployed.append(name)

    def clone_model(self, name: str):
        self.cloned.append(name)


class DummyRB:
    """Rollback manager missing auto_rollback."""
    pass


def test_rank_and_deploy():
    cmp = CrossModelComparator(pathways=None, history=DummyHistory(), deployer=DummyDeployer())
    best = cmp.rank_and_deploy()
    assert best == "A"
    assert cmp.deployer.deployed == ["A"]
    assert cmp.deployer.cloned == ["A"]


def test_weighted_best_model():
    p = DummyPathways({"A": 2.0, "B": 0.5})
    h = DummyHistory()
    cmp = CrossModelComparator(pathways=p, history=h, deployer=DummyDeployer())
    assert cmp.best_model() == "A"


def test_redeploy_on_change():
    h = DummyHistory({"A": 0.8, "B": 1.0})
    p = DummyPathways()
    d = DummyDeployer()
    cmp = CrossModelComparator(pathways=p, history=h, deployer=d)
    cmp.rank_and_deploy()
    assert d.deployed == ["B"]
    assert d.cloned == ["B"]
    h.weights = {"A": 1.2, "B": 0.5}
    cmp.rank_and_deploy()
    assert d.deployed == ["B", "A"]
    assert d.cloned == ["B", "A"]
    cmp.rank_and_deploy()
    assert d.deployed == ["B", "A"]
    assert d.cloned == ["B", "A"]


def test_evaluate_no_auto_rollback():
    cmp = CrossModelComparator(
        pathways=None,
        history=DummyHistory({"A": 0.4}),
        deployer=None,
        rollback_mgr=DummyRB(),
    )
    assert cmp.evaluate_and_rollback() == "A"

