import json
import math

from mvp_evaluator import evaluate_roi


class ExplodingStr:
    def __str__(self) -> str:
        raise RuntimeError("boom")


def assert_json_float(value: float) -> None:
    assert isinstance(value, float)
    assert math.isfinite(value)
    json.dumps(value)


def test_evaluate_roi_is_deterministic():
    stdout = "line1\nline2"
    stderr = ""
    first = evaluate_roi(stdout, stderr)
    second = evaluate_roi(stdout, stderr)
    third = evaluate_roi(stdout, stderr)
    assert first == second == third
    assert_json_float(first)


def test_evaluate_roi_monotonic_stdout_growth():
    stderr = ""
    empty = evaluate_roi("", stderr)
    one_line = evaluate_roi("ok", stderr)
    many_lines = evaluate_roi("ok\n" * 5, stderr)
    assert empty <= one_line <= many_lines
    assert_json_float(many_lines)


def test_evaluate_roi_monotonic_stderr_growth():
    stdout = "ok"
    empty = evaluate_roi(stdout, "")
    one_line = evaluate_roi(stdout, "error")
    many_lines = evaluate_roi(stdout, "error\n" * 5)
    assert empty >= one_line >= many_lines
    assert_json_float(many_lines)


def test_evaluate_roi_edge_cases():
    empty = evaluate_roi("", "")
    assert_json_float(empty)

    only_stderr = evaluate_roi("", "bad")
    assert_json_float(only_stderr)
    assert only_stderr <= empty

    large_stdout = "a" * 100000
    large_stderr = "b" * 100000
    large = evaluate_roi(large_stdout, large_stderr)
    assert_json_float(large)


def test_evaluate_roi_unexpected_inputs():
    coerced = evaluate_roi(None, 123)
    assert_json_float(coerced)

    sentinel = evaluate_roi(ExplodingStr(), "")
    assert sentinel == -1.0
