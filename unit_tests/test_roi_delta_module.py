from decimal import Decimal

from roi_delta import compute_roi_delta


def test_identical_keys_return_deterministic_deltas() -> None:
    before = {"alpha": 1.25, "beta": -3.5}
    after = {"alpha": 1.25, "beta": -3.5}

    response_one = compute_roi_delta(before, after)
    response_two = compute_roi_delta(before, after)

    assert response_one == response_two
    assert response_one["status"] == "ok"
    assert response_one["data"] == {
        "deltas": {"alpha": Decimal("0.00"), "beta": Decimal("0.0")},
        "total": Decimal("0.00"),
    }


def test_mismatched_schema_returns_structured_error() -> None:
    before = {"alpha": 1.0}
    after = {"beta": 2.0}

    response = compute_roi_delta(before, after)

    assert response["status"] == "error"
    assert response["data"] == {"deltas": {}, "total": Decimal("0")}
    assert response["errors"] == [
        {
            "type": "ValidationError",
            "error_type": "ValidationError",
            "message": "before_metrics and after_metrics must have identical keys.",
            "rule_id": None,
            "rule_index": None,
            "code": "invalid_schema",
            "details": {
                "code": "invalid_schema",
                "missing": ["alpha"],
                "extra": ["beta"],
            },
        }
    ]
    assert response["meta"]["keys"] == ["alpha", "beta"]
    assert response["meta"]["error_count"] == 1
