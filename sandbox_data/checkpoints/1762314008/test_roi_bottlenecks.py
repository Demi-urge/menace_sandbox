import yaml

from roi_calculator import ROICalculator, propose_fix


def _write_profiles(path):
    base_weights = {
        "profitability": 0.25,
        "efficiency": 0.2,
        "reliability": 0.15,
        "resilience": 0.1,
        "maintainability": 0.1,
        "security": 0.1,
        "latency": -0.05,
        "energy": -0.05,
    }
    profiles = {
        "scraper_bot": {
            "weights": base_weights,
            "veto": {
                "security": {"min": 0.4},
                "alignment_violation": {"equals": True},
            },
        }
    }
    path.write_text(yaml.safe_dump(profiles))
    return path


def test_propose_fix_prioritises_veto_and_lowest_contributors(tmp_path):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    profile = calc.profiles["scraper_bot"]
    metrics = {
        "profitability": 1.0,
        "efficiency": 1.0,
        "reliability": 1.0,
        "resilience": 1.0,
        "maintainability": 1.0,
        "security": 0.2,
        "latency": 1.0,
        "energy": 0.5,
    }
    fixes = propose_fix(metrics, profile)
    assert fixes[0] == (
        "security", "harden authentication; add input validation"
    )
    assert (
        "latency", "optimise I/O; use caching"
    ) in fixes
    assert (
        "energy", "batch work; reduce polling"
    ) in fixes


def test_propose_fix_returns_top_bottlenecks_with_hints():
    profile = {
        "weights": {
            "profitability": 0.4,
            "efficiency": 0.3,
            "maintainability": 0.2,
            "security": 0.1,
        }
    }
    metrics = {
        "profitability": 0.9,
        "efficiency": 0.2,
        "maintainability": 0.4,
        "security": 0.7,
    }
    fixes = propose_fix(metrics, profile)
    assert fixes == [
        ("efficiency", "optimise algorithms; reduce overhead"),
        ("security", "harden authentication; add input validation"),
        ("maintainability", "refactor for clarity; improve documentation"),
    ]
