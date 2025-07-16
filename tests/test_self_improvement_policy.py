import menace.self_improvement_policy as sip


def test_policy_update():
    policy = sip.SelfImprovementPolicy()
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0)
    before = policy.score(state)
    policy.update(state, 1.0)
    after = policy.score(state)
    assert after > before


def test_policy_persistence(tmp_path):
    path = tmp_path / "policy.pkl"
    policy = sip.SelfImprovementPolicy(path=path)
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0)
    policy.update(state, 2.0)
    policy2 = sip.SelfImprovementPolicy(path=path)
    assert policy2.score(state) == policy.score(state)
