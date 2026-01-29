import json
import math

import pytest

from mvp_evaluator import evaluate_roi


@pytest.mark.parametrize(
    "stdout, stderr",
    [
        ("", ""),
        ("", "only stderr"),
        ("x" * 10_000, ""),
        ("x" * 50_000, "y" * 25_000),
        (None, None),
        (b"bytes output", b"bytes error"),
    ],
)
def test_evaluate_roi_returns_finite_float(stdout, stderr) -> None:
    score = evaluate_roi(stdout, stderr)

    assert isinstance(score, float)
    assert math.isfinite(score)
    json.dumps(score)


def test_evaluate_roi_is_deterministic() -> None:
    stdout = "deterministic output"
    stderr = "deterministic error"

    first = evaluate_roi(stdout, stderr)
    second = evaluate_roi(stdout, stderr)
    third = evaluate_roi(stdout, stderr)

    assert first == second == third


def test_evaluate_roi_monotonic_in_stdout() -> None:
    stderr = "fixed stderr"

    baseline = evaluate_roi("", stderr)
    medium = evaluate_roi("a" * 10, stderr)
    large = evaluate_roi("a" * 100, stderr)

    assert baseline <= medium <= large


def test_evaluate_roi_monotonic_in_stderr() -> None:
    stdout = "fixed stdout"

    baseline = evaluate_roi(stdout, "")
    medium = evaluate_roi(stdout, "b" * 10)
    large = evaluate_roi(stdout, "b" * 100)

    assert baseline >= medium >= large


def test_evaluate_roi_edge_cases() -> None:
    assert evaluate_roi("", "") == 0.0

    stderr_only = evaluate_roi("", "error")
    assert stderr_only < 0.0

    large_stdout = "s" * 200_000
    large_stderr = "e" * 200_000

    large_score = evaluate_roi(large_stdout, "")
    mixed_score = evaluate_roi(large_stdout, large_stderr)

    assert large_score >= mixed_score
