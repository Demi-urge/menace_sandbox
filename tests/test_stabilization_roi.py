import decimal
from decimal import Decimal

import pytest

from menace_sandbox.stabilization.roi import compute_roi_delta


def test_identical_metrics_return_zero_delta() -> None:
    metrics = {"alpha": 1.5, "beta": -2.0}

    response = compute_roi_delta(metrics, metrics)

    assert response["status"] == "ok"
    assert response["data"]["deltas"] == {"alpha": Decimal("0.0"), "beta": Decimal("0.0")}
    assert response["data"]["total"] == Decimal("0.0")
    assert response["errors"] == []
    assert response["meta"]["keys"] == ["alpha", "beta"]


def test_mixed_values_compute_arithmetic_deltas() -> None:
    before = {"alpha": 1.0, "beta": -2.5, "gamma": 10.0}
    after = {"alpha": 3.0, "beta": -3.5, "gamma": 7.5}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    assert response["data"]["deltas"] == {
        "alpha": Decimal("2.0"),
        "beta": Decimal("-1.0"),
        "gamma": Decimal("-2.5"),
    }
    assert response["data"]["total"] == Decimal("-1.5")


def test_negative_and_large_values_compute_raw_deltas() -> None:
    before = {"alpha": -1_000_000.0, "beta": 10**12}
    after = {"alpha": -2_500_000.5, "beta": 10**12 + 3_333.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    assert response["data"]["deltas"] == {
        "alpha": Decimal("-1500000.5"),
        "beta": Decimal("3333.0"),
    }
    assert response["data"]["total"] == Decimal("-1496667.5")
    assert response["meta"]["keys"] == ["alpha", "beta"]


def test_empty_metrics_succeed_with_zero_totals() -> None:
    response = compute_roi_delta({}, {})

    assert response["status"] == "ok"
    assert response["data"]["deltas"] == {}
    assert response["data"]["total"] == Decimal("0")
    assert response["meta"]["count"] == 0


def test_schema_mismatch_missing_keys_returns_error() -> None:
    before = {"alpha": 1.0, "beta": 2.0}
    after = {"alpha": 1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["meta"]["count"] == 2
    assert response["meta"]["error_count"] == 1
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "invalid_schema",
            "message": "before_metrics and after_metrics must have identical keys.",
            "details": {"missing": ["beta"], "extra": []},
        }
    ]


def test_schema_mismatch_extra_keys_returns_error() -> None:
    before = {"alpha": 1.0}
    after = {"alpha": 1.0, "beta": 2.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["meta"]["keys"] == ["alpha", "beta"]
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "invalid_schema",
            "message": "before_metrics and after_metrics must have identical keys.",
            "details": {"missing": [], "extra": ["beta"]},
        }
    ]


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
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["meta"]["error_count"] == 1
    expected_field = "before_metrics" if not isinstance(before_metrics, dict) else "after_metrics"
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "invalid_schema",
            "message": f"{expected_field} must be a mapping.",
            "details": {"field": expected_field},
        }
    ]


@pytest.mark.parametrize(
    "value",
    ["oops", object(), True, None],
)
def test_non_numeric_values_return_non_numeric_value_error(value: object) -> None:
    before = {"alpha": 1.0}
    after = {"alpha": value}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "metric_type_error",
            "message": "Metric value must be an int, float, or Decimal (bool not allowed).",
            "details": {
                "key": "alpha",
                "source": "after_metrics",
                "value_repr": repr(value),
            },
        }
    ]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf")],
)
def test_non_finite_values_return_non_finite_value_error(value: object) -> None:
    before = {"alpha": 1.0}
    after = {"alpha": value}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "metric_value_error",
            "message": "Metric value must be a finite int, float, or Decimal (bool not allowed).",
            "details": {
                "key": "alpha",
                "source": "after_metrics",
                "value_repr": repr(value),
            },
        }
    ]


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_non_finite_values_in_before_metrics_error(value: float) -> None:
    before = {"alpha": value}
    after = {"alpha": 1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "metric_value_error",
            "message": "Metric value must be a finite int, float, or Decimal (bool not allowed).",
            "details": {
                "key": "alpha",
                "source": "before_metrics",
                "value_repr": repr(value),
            },
        }
    ]


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
    deltas = response["data"]["deltas"]
    assert response["data"]["total"] == sum(deltas.values(), Decimal("0"))
    assert deltas == {
        "alpha": Decimal("0.5"),
        "beta": Decimal("-3.0"),
        "gamma": Decimal("1.0"),
    }
    assert response["data"]["total"] == Decimal("-1.5")


def test_missing_keys_are_not_treated_as_zero() -> None:
    before = {"alpha": 1.0, "beta": 2.0}
    after = {"alpha": 1.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["meta"]["keys"] == ["alpha", "beta"]
    assert response["meta"]["error_count"] == 1


@pytest.mark.parametrize("value", [False, True])
def test_boolean_values_are_rejected(value: bool) -> None:
    before = {"alpha": 1.0}
    after = {"alpha": value}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "metric_type_error",
            "message": "Metric value must be an int, float, or Decimal (bool not allowed).",
            "details": {
                "key": "alpha",
                "source": "after_metrics",
                "value_repr": repr(value),
            },
        }
    ]


def test_large_values_are_deterministic_and_unscaled() -> None:
    before = {"alpha": 10**30, "beta": -10**25}
    after = {"alpha": 10**30 + 7, "beta": -10**25 - 12}

    result_one = compute_roi_delta(before, after)
    result_two = compute_roi_delta(before, after)

    assert result_one == result_two
    assert result_one["data"]["deltas"] == {
        "alpha": Decimal("7"),
        "beta": Decimal("-12"),
    }
    assert result_one["data"]["total"] == Decimal("-5")


def test_decimal_inputs_return_decimal_deltas() -> None:
    before = {"alpha": Decimal("1.25"), "beta": Decimal("2")}
    after = {"alpha": Decimal("2.75"), "beta": Decimal("5")}

    response = compute_roi_delta(before, after)

    assert response["status"] == "ok"
    assert response["data"]["deltas"] == {
        "alpha": Decimal("1.5"),
        "beta": Decimal("3"),
    }
    assert response["data"]["total"] == Decimal("4.5")


def test_overflow_in_delta_returns_error() -> None:
    before = {"alpha": Decimal("-9.99e9")}
    after = {"alpha": Decimal("9.99e9")}

    with decimal.localcontext() as context:
        context.Emax = 9
        context.traps[decimal.Overflow] = False
        response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "delta_value_error",
            "message": "Computed delta value must be finite.",
            "details": {"key": "alpha", "delta_repr": "Decimal('Infinity')"},
        }
    ]


def test_overflow_in_total_returns_error() -> None:
    before = {"alpha": Decimal("0"), "beta": Decimal("0")}
    after = {"alpha": Decimal("9.99e9"), "beta": Decimal("9.99e9")}

    with decimal.localcontext() as context:
        context.Emax = 9
        context.traps[decimal.Overflow] = False
        response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "RoiDeltaValidationError",
            "code": "total_value_error",
            "message": "Computed total delta must be finite.",
            "details": {"total_delta_repr": "Decimal('Infinity')"},
        }
    ]
