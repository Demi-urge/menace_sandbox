import pytest
from sandbox_settings import SandboxSettings


def test_baseline_window_within_range():
    assert SandboxSettings(baseline_window=5).baseline_window == 5
    assert SandboxSettings(baseline_window=10).baseline_window == 10


def test_baseline_window_out_of_range():
    with pytest.raises(ValueError):
        SandboxSettings(baseline_window=4)
    with pytest.raises(ValueError):
        SandboxSettings(baseline_window=11)
