import math

from roi_calculator import ROICalculator


def test_default_roi_profile_scores_finite():
    calc = ROICalculator()  # uses default configs/roi_profiles.yaml
    assert "scraper_bot" in calc.profiles

    metrics = {
        "profitability": 1.0,
        "efficiency": 1.0,
        "reliability": 1.0,
        "resilience": 1.0,
        "maintainability": 1.0,
        "security": 0.6,  # meets veto threshold of min 0.5
        "latency": 0.0,
        "energy": 0.0,
        "alignment_violation": False,  # avoids equals true veto
    }

    result = calc.calculate(metrics, "scraper_bot")
    assert math.isfinite(result.score)
    assert not result.vetoed
    assert result.triggers == []
