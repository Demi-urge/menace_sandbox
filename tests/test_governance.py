import menace.governance as governance


def test_ship_veto_on_alignment_failure():
    rules = governance.load_rules()
    scorecard = {"decision": "ship", "alignment": "fail", "raroi_increase": 0}
    assert governance.check_veto(scorecard, rules)


def test_rollback_veto_on_raroi_increase():
    rules = governance.load_rules()
    scorecard = {"decision": "rollback", "alignment": "pass", "raroi_increase": 3}
    msgs = governance.check_veto(scorecard, rules)
    assert msgs and "rollback" in msgs[0]


def test_rule_pass_when_conditions_not_met():
    rules = governance.load_rules()
    scorecard = {"decision": "ship", "alignment": "pass", "raroi_increase": 0}
    assert not governance.check_veto(scorecard, rules)

