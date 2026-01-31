import math
from decimal import Decimal

from roi_delta import compute_roi_delta


def test_empty_dicts() -> None:
    result = compute_roi_delta({}, {})

    assert result["status"] == "ok"
    assert result["data"] == {"deltas": {}, "total": Decimal("0")}
    assert result["errors"] == []
    assert result["meta"]["count"] == 0
    assert result["meta"]["keys"] == []


def test_mismatched_keys() -> None:
    result = compute_roi_delta({"a": 1.0}, {"b": 1.0})

    assert result["status"] == "error"
    assert result["data"]["deltas"] == {}
    assert result["errors"][0]["type"] == "ValidationError"
    assert result["errors"][0]["code"] == "invalid_schema"
    assert result["errors"][0]["details"]["missing"] == ["a"]
    assert result["errors"][0]["details"]["extra"] == ["b"]


def test_invalid_schema_types() -> None:
    result = compute_roi_delta([("a", 1.0)], {"a": 1.0})

    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "invalid_schema"


def test_non_numeric_values_fail() -> None:
    result = compute_roi_delta({"a": 1.0}, {"a": "bad"})

    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "metric_type_error"


def test_bool_values_fail() -> None:
    result = compute_roi_delta({"a": 1.0}, {"a": True})

    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "metric_type_error"


def test_nan_and_inf_values_fail() -> None:
    for value in [math.nan, math.inf, -math.inf]:
        result = compute_roi_delta({"a": 1.0}, {"a": value})

        assert result["status"] == "error"
        assert result["errors"][0]["code"] == "metric_value_error"


def test_negative_values() -> None:
    result = compute_roi_delta({"a": -5.0}, {"a": -10.0})

    assert result["status"] == "ok"
    assert result["data"]["deltas"] == {"a": Decimal("-5")}
    assert result["data"]["total"] == Decimal("-5")


def test_determinism() -> None:
    before = {"a": 1.5, "b": 2.5}
    after = {"a": 2.5, "b": 5.5}

    result_one = compute_roi_delta(before, after)
    result_two = compute_roi_delta(before, after)

    assert result_one == result_two
