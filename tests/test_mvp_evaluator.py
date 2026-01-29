import json
import math

from mvp_evaluator import evaluate_mvp_roi


class ExplodingStr:
    def __str__(self) -> str:
        raise RuntimeError("boom")


def assert_finite_float(value: float) -> None:
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_evaluate_mvp_roi_deterministic():
    stdout = "ok\nline"
    stderr = "warn"
    first = evaluate_mvp_roi(stdout, stderr)
    second = evaluate_mvp_roi(stdout, stderr)
    assert first == second


def test_evaluate_mvp_roi_monotonicity():
    base_stdout = "success"
    base_stderr = ""

    low_stderr = evaluate_mvp_roi(base_stdout, "err")
    high_stderr = evaluate_mvp_roi(base_stdout, "err" * 10)
    assert high_stderr <= low_stderr

    low_stdout = evaluate_mvp_roi("ok", base_stderr)
    high_stdout = evaluate_mvp_roi("ok" * 10, base_stderr)
    assert high_stdout >= low_stdout


def test_evaluate_mvp_roi_edge_cases_and_errors():
    assert_finite_float(evaluate_mvp_roi("", ""))
    assert_finite_float(evaluate_mvp_roi("", "only stderr"))

    huge_stdout = "a" * 200_000
    huge_stderr = "b" * 150_000
    assert_finite_float(evaluate_mvp_roi(huge_stdout, huge_stderr))

    assert evaluate_mvp_roi(ExplodingStr(), "stderr") == -1.0
    assert evaluate_mvp_roi("stdout", ExplodingStr()) == -1.0


def test_evaluate_mvp_roi_json_serializable():
    value = evaluate_mvp_roi("output", "")
    assert_finite_float(value)
    dumped = json.dumps({"roi": value})
    assert "roi" in dumped
