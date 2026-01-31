import math

from roi_delta import compute_roi_delta


def test_empty_dicts() -> None:
    result = compute_roi_delta({}, {})

    assert result["status"] == "ok"
    assert result["data"] == {"deltas": {}, "total_delta": 0.0}
    assert result["errors"] == []
    assert result["meta"]["metric_count"] == 0
    assert isinstance(result["meta"]["input_hash"], str)


def test_mismatched_keys() -> None:
    result = compute_roi_delta({"a": 1.0}, {"b": 1.0})

    assert result["status"] == "failed"
    assert result["data"] == {}
    assert result["errors"][0]["type"] == "ValidationError"
    assert result["errors"][0]["details"]["missing"] == ["a"]
    assert result["errors"][0]["details"]["extra"] == ["b"]


def test_nan_and_inf_values_fail() -> None:
    for value in [math.nan, math.inf, -math.inf]:
        result = compute_roi_delta({"a": 1.0}, {"a": value})

        assert result["status"] == "failed"
        assert result["data"] == {}
        assert result["errors"][0]["type"] == "ValidationError"


def test_negative_values() -> None:
    result = compute_roi_delta({"a": -5.0}, {"a": -10.0})

    assert result["status"] == "ok"
    assert result["data"]["deltas"] == {"a": -5.0}
    assert result["data"]["total_delta"] == -5.0


def test_extremely_large_values() -> None:
    before = {"a": 1.0e308, "b": -1.0e308}
    after = {"a": 1.0e308 + 1.0e292, "b": -1.0e308 + 1.0e292}

    result = compute_roi_delta(before, after)

    assert result["status"] == "ok"
    assert result["data"]["deltas"]["a"] == after["a"] - before["a"]
    assert result["data"]["deltas"]["b"] == after["b"] - before["b"]


def test_determinism() -> None:
    before = {"a": 1.5, "b": 2.5}
    after = {"a": 2.5, "b": 5.5}

    result_one = compute_roi_delta(before, after)
    result_two = compute_roi_delta(before, after)

    assert result_one == result_two
