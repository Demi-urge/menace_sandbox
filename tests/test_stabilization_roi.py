import math

import pytest

from menace_sandbox.stabilization.roi import compute_roi_delta


def _error_codes(response: dict) -> list[str]:
    return [error["code"] for error in response["errors"]]


def test_identical_metrics_return_zero_delta() -> None:
    metrics = {"alpha": 1.5, "beta": -2.0}

    response = compute_roi_delta(metrics, metrics)

    assert response["status"] == "ok"
    assert response["data"]["delta"] == {"alpha": 0.0, "beta": 0.0}


def test_mixed_values_compute_arithmetic_deltas() -> None:
    before = {"alpha": 1.0, "beta": -2.5, "gamma": 10.0}
    after = {"alpha": 3.0, "beta": -3.5, "gamma": 7.5}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    assert response["data"]["delta"] == {"alpha": 2.0, "beta": -1.0, "gamma": -2.5}


def test_negative_and_large_values_compute_raw_deltas() -> None:
    before = {"alpha": -1_000_000.0, "beta": 10**12}
    after = {"alpha": -2_500_000.5, "beta": 10**12 + 3_333.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    assert response["data"]["delta"] == {"alpha": -1_500_000.5, "beta": 3333.0}
    assert response["meta"]["input_keys"] == ["alpha", "beta"]


def test_empty_metrics_succeed_with_zero_totals() -> None:
    response = compute_roi_delta({}, {})

    assert response["status"] == "ok"
    assert response["data"]["delta"] == {}
    assert response["meta"]["key_count"] == 0


def test_schema_mismatch_missing_keys_returns_error() -> None:
    before = {"alpha": 1.0, "beta": 2.0}
    after = {"alpha": 1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert "invalid_schema" in _error_codes(response)
    assert response["data"] == {}


def test_schema_mismatch_extra_keys_returns_error() -> None:
    before = {"alpha": 1.0}
    after = {"alpha": 1.0, "beta": 2.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert "invalid_schema" in _error_codes(response)
    assert response["data"] == {}


@pytest.mark.parametrize(
    ("before_metrics", "after_metrics"),
    [
        ([], {"alpha": 1.0}),
        ({"alpha": 1.0}, None),
    ],
)
def test_non_mapping_inputs_return_input_type_error(
    before_metrics: object, after_metrics: object
) -> None:
    response = compute_roi_delta(before_metrics, after_metrics)

    assert response["status"] == "error"
    assert "invalid_schema" in _error_codes(response)
    assert response["data"] == {}


@pytest.mark.parametrize(
    "value",
    ["oops", object(), True, None],
)
def test_non_numeric_values_return_non_numeric_value_error(value: object) -> None:
    before = {"alpha": 1.0}
    after = {"alpha": value}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert "metric_type_error" in _error_codes(response)
    assert response["data"] == {}


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), 10**1000],
)
def test_non_finite_values_return_non_finite_value_error(value: object) -> None:
    before = {"alpha": 1.0}
    after = {"alpha": value}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert "metric_value_error" in _error_codes(response)
    assert response["data"] == {}


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_non_finite_values_in_before_metrics_error(value: float) -> None:
    before = {"alpha": value}
    after = {"alpha": 1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert "metric_value_error" in _error_codes(response)
    assert response["data"] == {}


def test_compute_roi_delta_is_deterministic() -> None:
    before = {"alpha": 2.0, "beta": 5.0}
    after = {"alpha": 1.0, "beta": 7.0}

    results = [compute_roi_delta(before, after) for _ in range(3)]

    assert results[0] == results[1] == results[2]


def test_delta_sum_matches_expected_per_key_deltas() -> None:
    before = {"alpha": 1.0, "beta": 4.0, "gamma": -2.0}
    after = {"alpha": 1.5, "beta": 1.0, "gamma": -1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    deltas = response["data"]["delta"]
    assert math.isclose(sum(deltas.values()), -2.5)
