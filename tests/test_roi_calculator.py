import yaml
import pytest

import yaml
import pytest

from menace_sandbox.roi_calculator import ROICalculator


def _write_profiles(path):
    profiles = {
        "scraper_bot": {
            "weights": {
                "profitability": 0.3,
                "efficiency": 0.2,
                "reliability": 0.1,
                "resilience": 0.1,
                "maintainability": 0.1,
                "security": 0.1,
                "latency": -0.1,
                "energy": -0.05,
            },
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
    score, vetoed, triggers = calc.calculate(metrics, "scraper_bot")
    assert score == pytest.approx(0.75)
    assert vetoed is False
    assert triggers == []


@pytest.mark.parametrize("metrics,trigger", [
    ({"security": 0.2}, "security"),
    ({"alignment_violation": True}, "alignment_violation"),
])
def test_veto_triggers(tmp_path, metrics, trigger):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    score, vetoed, triggers = calc.calculate(metrics, "scraper_bot")
    assert score == float("-inf")
    assert vetoed is True
    assert any(trigger in t for t in triggers)


def test_log_debug_outputs_components_and_veto(tmp_path, capsys):
    profile_path = _write_profiles(tmp_path / "roi_profiles.yaml")
    calc = ROICalculator(profiles_path=profile_path)
    metrics = {"profitability": 1, "security": 0.2}
    calc.log_debug(metrics, "scraper_bot")
    out = capsys.readouterr().out
    assert "profitability * 0.3 = 0.3" in out
    assert "Final score: -inf" in out
    assert "Veto triggers" in out
