import decimal
import math
import unittest
from decimal import Decimal

from menace_sandbox.stabilization.roi import compute_roi_delta


class TestRoiDeltaModule(unittest.TestCase):
    def test_success_path_with_deterministic_keys_and_deltas(self) -> None:
        before = {"b": 1.5, "a": 2.0}
        after = {"b": 2.5, "a": 1.0}

        response = compute_roi_delta(before, after)

        self.assertEqual(response["status"], "ok")
        self.assertEqual(
            response["data"],
            {"deltas": {"a": Decimal("-1.0"), "b": Decimal("1.0")}, "total": Decimal("0.0")},
        )
        self.assertEqual(response["errors"], [])
        self.assertEqual(
            response["meta"],
            {
                "keys": ["b", "a"],
                "count": 2,
                "error_count": 0,
                "before_count": 2,
                "after_count": 2,
            },
        )

    def test_empty_dicts_return_ok_with_empty_metadata(self) -> None:
        response = compute_roi_delta({}, {})

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["data"], {"deltas": {}, "total": Decimal("0")})
        self.assertEqual(response["errors"], [])
        self.assertEqual(
            response["meta"],
            {
                "keys": [],
                "count": 0,
                "error_count": 0,
                "before_count": 0,
                "after_count": 0,
            },
        )

    def test_schema_mismatch_returns_typed_error(self) -> None:
        before = {"alpha": 1.0, "beta": 2.0}
        after = {"beta": 2.0, "gamma": 3.0}

        response = compute_roi_delta(before, after)

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["data"], {"deltas": {}, "total": Decimal("0")})
        self.assertEqual(response["meta"]["keys"], ["alpha", "beta", "gamma"])
        self.assertEqual(response["meta"]["count"], 3)
        self.assertEqual(response["meta"]["error_count"], 1)
        self.assertEqual(response["errors"], [
            {
                "type": "RoiDeltaValidationError",
                "code": "invalid_schema",
                "message": "before_metrics and after_metrics must have identical keys.",
                "details": {"missing": ["alpha"], "extra": ["gamma"]},
            }
        ])

    def test_non_numeric_and_non_finite_values_rejected(self) -> None:
        cases = [
            ("oops", "after_metrics"),
            (None, "after_metrics"),
            (True, "after_metrics"),
            (math.nan, "after_metrics"),
            (math.inf, "after_metrics"),
            (-math.inf, "after_metrics"),
        ]
        for value, source in cases:
            with self.subTest(value=value):
                response = compute_roi_delta({"alpha": 1.0}, {"alpha": value})

                self.assertEqual(response["status"], "error")
                self.assertEqual(response["data"], {"deltas": {}, "total": Decimal("0")})
                self.assertEqual(response["meta"]["error_count"], 1)
                self.assertEqual(response["errors"], [
                    {
                        "type": "RoiDeltaValidationError",
                        "code": "metric_type_error" if not isinstance(value, (int, float)) or isinstance(value, bool) else "metric_value_error",
                        "message": (
                            "Metric value must be an int, float, or Decimal (bool not allowed)."
                            if not isinstance(value, (int, float)) or isinstance(value, bool)
                            else "Metric value must be a finite int, float, or Decimal (bool not allowed)."
                        ),
                        "details": {
                            "key": "alpha",
                            "source": source,
                            "value_repr": repr(value),
                        },
                    }
                ])

        response = compute_roi_delta({"alpha": math.nan}, {"alpha": 1.0})
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["errors"], [
            {
                "type": "RoiDeltaValidationError",
                "code": "metric_value_error",
                "message": "Metric value must be a finite int, float, or Decimal (bool not allowed).",
                "details": {
                    "key": "alpha",
                    "source": "before_metrics",
                    "value_repr": repr(math.nan),
                },
            }
        ])

    def test_negative_and_large_values_are_exact(self) -> None:
        before = {"alpha": -10**20, "beta": 10**30}
        after = {"alpha": -10**20 - 5, "beta": 10**30 + 7}

        response = compute_roi_delta(before, after)

        self.assertEqual(response["status"], "ok")
        self.assertEqual(
            response["data"],
            {"deltas": {"alpha": Decimal("-5"), "beta": Decimal("7")}, "total": Decimal("2")},
        )

    def test_determinism_and_missing_data_defaults(self) -> None:
        before = {"alpha": 2.0, "beta": 5.0}
        after = {"alpha": 1.0, "beta": 7.0}

        result_one = compute_roi_delta(before, after)
        result_two = compute_roi_delta(before, after)

        self.assertEqual(result_one, result_two)

        missing_response = compute_roi_delta({"alpha": 1.0, "beta": 2.0}, {"alpha": 1.0})
        self.assertEqual(missing_response["status"], "error")
        self.assertEqual(missing_response["data"], {"deltas": {}, "total": Decimal("0")})
        self.assertEqual(missing_response["meta"]["before_count"], 2)
        self.assertEqual(missing_response["meta"]["after_count"], 1)

    def test_overflow_in_delta_returns_error(self) -> None:
        before = {"alpha": Decimal("-9.99e9")}
        after = {"alpha": Decimal("9.99e9")}

        with decimal.localcontext() as context:
            context.Emax = 9
            context.traps[decimal.Overflow] = False
            response = compute_roi_delta(before, after)

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["data"], {"deltas": {}, "total": Decimal("0")})
        self.assertEqual(response["errors"], [
            {
                "type": "RoiDeltaValidationError",
                "code": "delta_value_error",
                "message": "Computed delta value must be finite.",
                "details": {"key": "alpha", "delta_repr": "Decimal('Infinity')"},
            }
        ])

    def test_overflow_in_total_returns_error(self) -> None:
        before = {"alpha": Decimal("0"), "beta": Decimal("0")}
        after = {"alpha": Decimal("9.99e9"), "beta": Decimal("9.99e9")}

        with decimal.localcontext() as context:
            context.Emax = 9
            context.traps[decimal.Overflow] = False
            response = compute_roi_delta(before, after)

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["data"], {"deltas": {}, "total": Decimal("0")})
        self.assertEqual(response["errors"], [
            {
                "type": "RoiDeltaValidationError",
                "code": "total_value_error",
                "message": "Computed total delta must be finite.",
                "details": {"total_delta_repr": "Decimal('Infinity')"},
            }
        ])


if __name__ == "__main__":
    unittest.main()
