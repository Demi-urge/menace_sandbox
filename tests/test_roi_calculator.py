"""Tests for :mod:`roi_calculator`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from roi_calculator import ROICalculator


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "roi_profiles.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    PROFILES = yaml.safe_load(fh)


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
def test_weighted_roi(profile: str) -> None:
    calc = ROICalculator()
    metrics = dict(BASE_METRICS)
    score = calc.compute(metrics, profile, {"alignment_violation": False})
    expected = sum(PROFILES[profile]["metrics"].values())
    assert score == pytest.approx(expected)
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
def test_veto_yields_neg_inf(
    profile: str, metrics: dict[str, float], flags: dict[str, bool]
) -> None:
    calc = ROICalculator()
    result = calc.compute(metrics, profile, flags)
    assert result == float("-inf")
    assert calc.hard_fail


def test_log_debug_breakdown(caplog: pytest.LogCaptureFixture) -> None:
    calc = ROICalculator()
    metrics = dict(BASE_METRICS)
    profile = "scraper_bot"
    flags = {"alignment_violation": False}
    weights = PROFILES[profile]["metrics"]
    contributions = {name: metrics[name] * weight for name, weight in weights.items()}
    score = sum(contributions.values())

    caplog.set_level(logging.DEBUG)
    calc.log_debug(metrics, profile, flags)
    messages = [rec.getMessage() for rec in caplog.records]

    assert f"weights: {weights}" in messages
    assert f"metrics: {metrics}" in messages
    assert f"flags: {flags}" in messages
    assert f"contributions: {contributions}" in messages
    assert f"final_score: {score}" in messages
    assert "veto_triggered: []" in messages

