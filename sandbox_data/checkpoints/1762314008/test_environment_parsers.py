import pytest
import sandbox_runner.environment as env


@pytest.mark.parametrize(
    "value,expected",
    [
        ("100kbps", 100_000),
        ("2mbps", 2_000_000),
        ("1gbps", 1_000_000_000),
        ("bad", 0),
        (123, 123),
    ],
)
def test_parse_bandwidth(value, expected):
    assert env._parse_bandwidth(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("500ms", 0.5),
        ("10s", 10.0),
        ("2m", 120.0),
        ("1h", 3600.0),
        ("1d", 86400.0),
        ("1w", 604800.0),
        ("bad", 0.0),
        (60, 60.0),
    ],
)
def test_parse_timespan(value, expected):
    assert env._parse_timespan(value) == pytest.approx(expected)


@pytest.mark.parametrize("cpu", ["0", "65", "-1", "foo"])
def test_validate_preset_bad_cpu(cpu):
    assert not env.validate_preset({"CPU_LIMIT": cpu})


@pytest.mark.parametrize("mem", ["0", "65gi", "foo"])
def test_validate_preset_bad_memory(mem):
    assert not env.validate_preset({"MEMORY_LIMIT": mem})


@pytest.mark.parametrize("key", ["BANDWIDTH_LIMIT", "MIN_BANDWIDTH", "MAX_BANDWIDTH"])
@pytest.mark.parametrize("value", ["0kbps", "20gbps", "foo"])
def test_validate_preset_bad_bandwidth(key, value):
    assert not env.validate_preset({key: value})


def test_validate_preset_valid():
    preset = {
        "CPU_LIMIT": "4",
        "MEMORY_LIMIT": "4gi",
        "BANDWIDTH_LIMIT": "100kbps",
    }
    assert env.validate_preset(preset)
