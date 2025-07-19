import self_improvement_policy as sip


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


def test_policy_adaptive_hyperparams():
    policy = sip.SelfImprovementPolicy(alpha=0.5, gamma=0.9, adaptive=True)
    state = (1,) * 15
    for _ in range(5):
        policy.update(state, 1.0)
    assert policy.alpha < 0.5
    assert policy.gamma <= 0.9


def test_policy_action_selection():
    policy = sip.SelfImprovementPolicy(epsilon=0.0)
    state = (2,) * 15
    policy.update(state, 1.0)
    assert policy.select_action(state) == 1
    policy = sip.SelfImprovementPolicy(epsilon=1.0)
    counts = {0: 0, 1: 0}
    for _ in range(50):
        counts[policy.select_action(state)] += 1
    assert counts[0] > 0 and counts[1] > 0
