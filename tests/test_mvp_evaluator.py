import json
import math

import pytest

from mvp_evaluator import evaluate_mvp_roi


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
def test_evaluate_mvp_roi_returns_finite_float(stdout, stderr) -> None:
    score = evaluate_mvp_roi(stdout, stderr)

    assert isinstance(score, float)
    assert math.isfinite(score)
    json.dumps(score)


def test_evaluate_mvp_roi_is_deterministic() -> None:
    stdout = "deterministic output"
    stderr = "deterministic error"

    first = evaluate_mvp_roi(stdout, stderr)
    second = evaluate_mvp_roi(stdout, stderr)
    third = evaluate_mvp_roi(stdout, stderr)

    assert first == second == third


def test_evaluate_mvp_roi_monotonic_in_stdout() -> None:
    stderr = "fixed stderr"

    baseline = evaluate_mvp_roi("", stderr)
    medium = evaluate_mvp_roi("a" * 10, stderr)
    large = evaluate_mvp_roi("a" * 100, stderr)

    assert baseline <= medium <= large


def test_evaluate_mvp_roi_monotonic_in_stderr() -> None:
    stdout = "fixed stdout"

    baseline = evaluate_mvp_roi(stdout, "")
    medium = evaluate_mvp_roi(stdout, "b" * 10)
    large = evaluate_mvp_roi(stdout, "b" * 100)

    assert baseline >= medium >= large


def test_evaluate_mvp_roi_edge_cases() -> None:
    assert evaluate_mvp_roi("", "") == 0.0

    stderr_only = evaluate_mvp_roi("", "error")
    assert stderr_only < 0.0

    large_stdout = "s" * 200_000
    large_stderr = "e" * 200_000

    large_score = evaluate_mvp_roi(large_stdout, "")
    mixed_score = evaluate_mvp_roi(large_stdout, large_stderr)

    assert large_score >= mixed_score
