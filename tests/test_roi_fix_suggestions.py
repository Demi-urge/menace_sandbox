from roi_calculator import propose_fix


def test_propose_fix_returns_top_bottleneck_hints():
    profile = {
        "weights": {
            "profitability": 0.5,
            "efficiency": 0.2,
            "security": 0.2,
            "maintainability": 0.1,
        }
    }
    metrics = {
        "profitability": 0.9,
        "efficiency": 0.1,
        "security": 0.3,
        "maintainability": 0.4,
    }
    fixes = propose_fix(metrics, profile)
    assert fixes == [
        ("efficiency", "optimise algorithms; reduce overhead"),
        ("maintainability", "refactor for clarity; improve documentation"),
        ("security", "harden authentication; add input validation"),
    ]
