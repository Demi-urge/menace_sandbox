import logging
import pytest

from roi_calculator import ROICalculator

BASE_METRICS = {
    "profitability": 1.0,
    "efficiency": 1.0,
    "reliability": 1.0,
    "resilience": 1.0,
    "maintainability": 1.0,
    "security": 1.0,
    "latency": 1.0,
    "energy": 1.0,
}


@pytest.mark.parametrize("profile", ["scraper_bot", "content_generator", "growth_planner"])
def test_weighted_roi(profile):
    calc = ROICalculator(profile)
    metrics = dict(BASE_METRICS)
    score = calc.calculate(metrics, {"alignment_violation": False})
    assert score == pytest.approx(sum(calc.weights.values()))
    assert not calc.hard_fail


@pytest.mark.parametrize(
    "profile,metrics,flags",
    [
        ("scraper_bot", {**BASE_METRICS, "security": 0.0}, {"alignment_violation": False}),
        ("content_generator", {**BASE_METRICS, "security": 0.0}, {"alignment_violation": False}),
        ("growth_planner", {**BASE_METRICS, "security": 0.0}, {"alignment_violation": False}),
        ("scraper_bot", dict(BASE_METRICS), {"alignment_violation": True}),
        ("content_generator", dict(BASE_METRICS), {"alignment_violation": True}),
        ("growth_planner", dict(BASE_METRICS), {"alignment_violation": True}),
    ],
)
def test_veto_yields_neg_inf(profile, metrics, flags):
    calc = ROICalculator(profile)
    result = calc.calculate(metrics, flags)
    assert result == float("-inf")
    assert calc.hard_fail


def test_log_debug_breakdown(caplog):
    calc = ROICalculator("scraper_bot")
    metrics = dict(BASE_METRICS)
    flags = {"alignment_violation": False}
    contributions = {name: metrics[name] * weight for name, weight in calc.weights.items()}
    score = sum(contributions.values())

    caplog.set_level(logging.DEBUG)
    calc.log_debug(metrics, flags)
    messages = [rec.getMessage() for rec in caplog.records]

    assert f"weights: {calc.weights}" in messages
    assert f"metrics: {metrics}" in messages
    assert f"flags: {flags}" in messages
    assert f"contributions: {contributions}" in messages
    assert f"final_score: {score}" in messages
    assert "veto_triggered: []" in messages
