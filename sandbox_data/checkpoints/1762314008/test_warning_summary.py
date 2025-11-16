"""Tests for ROI warning summary reporting."""

import logging

import menace.self_improvement as sie
from human_alignment_flagger import flag_improvement


def test_positive_roi_with_high_severity_warning_recorded() -> None:
    """Positive ROI with risk warnings should be summarised."""
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.logger = logging.getLogger("test")
    engine.warning_summary = []

    warnings = flag_improvement(
        None,
        {"actions": [{"reward": 100.0, "risk_score": 8.0}]},
        None,
    )

    engine._record_warning_summary(0.5, warnings)

    assert engine.warning_summary
    entry = engine.warning_summary[0]
    assert entry["roi_delta"] == 0.5
    assert entry["warnings"]["risk_reward"]

