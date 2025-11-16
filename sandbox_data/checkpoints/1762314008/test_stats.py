import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.stats as stats


def test_wilson_interval():
    lower, upper = stats.wilson_score_interval(9, 10)
    assert lower > 0.7 and upper <= 1.0
