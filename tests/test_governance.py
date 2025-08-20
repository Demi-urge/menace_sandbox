import governance


def test_alignment_failure_blocks_shipping():
    scorecards = {"normal": {"roi_delta": 0.1}}
    allow_ship, allow_rollback, reasons = governance.evaluate_rules(
        scorecards, "fail", [0.0, -0.1]
    )
    assert not allow_ship
    assert allow_rollback
    assert any("alignment failure" in r for r in reasons)


def test_raroi_increase_blocks_rollback():
    scorecards = {"normal": {"roi_delta": 0.1}}
    allow_ship, allow_rollback, reasons = governance.evaluate_rules(
        scorecards, "pass", [0.2, 0.1, 0.3, -0.1]
    )
    assert allow_ship
    assert not allow_rollback
    assert any("RAROI increased" in r for r in reasons)
