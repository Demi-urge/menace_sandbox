import menace.governance as governance


def test_ship_veto_on_alignment_failure():
    assert governance.evaluate_governance("ship", "fail", [])


def test_rollback_veto_on_raroi_increase():
    deltas = [0.1, -0.2, 0.05, 0.3]
    msgs = governance.evaluate_governance("rollback", "pass", deltas)
    assert msgs and "rollback" in msgs[0]


def test_register_rule():
    def extra(decision, *_):
        if decision == "ship":
            return ["extra veto"]
        return []

    governance.register_rule(extra)
    msgs = governance.evaluate_governance("ship", "pass", [])
    assert "extra veto" in msgs
    governance._EXTRA_RULES.clear()
