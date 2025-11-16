from roi_calculator import propose_fix


def test_ranking_logic_for_multiple_metrics():
    profile = {
        "weights": {
            "profitability": 0.5,
            "efficiency": 0.3,
            "security": 0.2,
        }
    }
    metrics = {
        "profitability": 0.9,
        "efficiency": 0.2,
        "security": 0.4,
    }
    fixes = propose_fix(metrics, profile)
    assert fixes == [
        ("efficiency", "optimise algorithms; reduce overhead"),
        ("security", "harden authentication; add input validation"),
        ("profitability", "optimise revenue streams; reduce costs"),
    ]


def test_known_metric_hint_generation():
    profile = {
        "weights": {
            "reliability": 0.6,
            "efficiency": 0.4,
        }
    }
    metrics = {
        "reliability": 0.1,
        "efficiency": 0.9,
    }
    fixes = propose_fix(metrics, profile)
    assert fixes[0] == (
        "reliability",
        "increase retries; improve test mocks",
    )


def test_unknown_metric_and_profile_handling():
    profile = {"weights": {"mystery": 1.0}}
    metrics = {"mystery": 0.2}
    fixes = propose_fix(metrics, profile)
    assert fixes == [("mystery", "improve mystery")]
    assert propose_fix(metrics, "nonexistent_profile") == []
