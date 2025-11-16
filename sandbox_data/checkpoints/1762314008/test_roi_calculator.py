import logging

import pytest
import yaml

from menace_sandbox.roi_calculator import ROICalculator


def _write_profiles(path, weights=None):
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
    if weights is not None:
        base_weights = weights
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


def test_weighted_roi(tmp_path):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    metrics = {
        "profitability": 1,
        "efficiency": 1,
        "reliability": 1,
        "resilience": 1,
        "maintainability": 1,
        "security": 1,
        "latency": 1,
        "energy": 1,
    }
    result = calc.calculate(metrics, "scraper_bot")
    assert result.score == pytest.approx(0.8)
    assert result.vetoed is False
    assert result.triggers == []


@pytest.mark.parametrize("metrics,trigger", [
    ({"security": 0.2}, "security"),
    ({"alignment_violation": True}, "alignment_violation"),
])
def test_veto_triggers(tmp_path, metrics, trigger):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    result = calc.calculate(metrics, "scraper_bot")
    assert result.score == float("-inf")
    assert result.vetoed is True
    assert any(trigger in t for t in result.triggers)


def test_log_debug_outputs_components_and_veto(tmp_path, caplog):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    metrics = {"profitability": 1, "security": 0.2}
    with caplog.at_level(logging.DEBUG, logger="menace_sandbox.roi_calculator"):
        calc.log_debug(metrics, "scraper_bot")
    assert "profitability * 0.25 = 0.25" in caplog.text
    assert "Final score: -inf" in caplog.text
    assert "Veto triggers" in caplog.text


def test_unknown_profile_raises(tmp_path):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    with pytest.raises(ValueError):
        calc.calculate({}, "unknown")


def test_profile_missing_metric_raises(tmp_path):
    weights = {
        "profitability": 1,
        "efficiency": 0,
        "reliability": 0,
        "resilience": 0,
        "maintainability": 0,
        "security": 0,
        "latency": 0,
        # 'energy' omitted
    }
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml", weights)
    with pytest.raises(ValueError):
        ROICalculator(profiles_path=profile_path)


def test_profile_invalid_weight_sum_raises(tmp_path):
    weights = {
        "profitability": 0.5,
        "efficiency": 0,
        "reliability": 0,
        "resilience": 0,
        "maintainability": 0,
        "security": 0,
        "latency": 0,
        "energy": 0.6,
    }
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml", weights)
    with pytest.raises(ValueError):
        ROICalculator(profiles_path=profile_path)


def test_profile_weight_sum_too_low_raises(tmp_path):
    weights = {
        "profitability": 0.2,
        "efficiency": 0.1,
        "reliability": 0.1,
        "resilience": 0.1,
        "maintainability": 0.1,
        "security": 0.1,
        "latency": 0.1,
        "energy": 0.1,
    }
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml", weights)
    with pytest.raises(ValueError):
        ROICalculator(profiles_path=profile_path)


def test_profile_negative_weights_abs_sum_valid(tmp_path):
    weights = {
        "profitability": 0.5,
        "efficiency": 0,
        "reliability": 0,
        "resilience": 0,
        "maintainability": 0,
        "security": 0,
        "latency": 0,
        "energy": -0.5,
    }
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml", weights)
    calc = ROICalculator(profiles_path=profile_path)
    metrics = {"profitability": 1, "energy": 1, "security": 0.5}
    result = calc.calculate(metrics, "scraper_bot")
    assert result.score == 0.0
    assert result.vetoed is False
    assert result.triggers == []
