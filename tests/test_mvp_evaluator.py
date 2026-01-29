import json

from mvp_evaluator import evaluate_roi


def test_evaluate_roi_deterministic():
    stdout = "success" * 3
    stderr = "warning" * 2

    first = evaluate_roi(stdout, stderr)
    second = evaluate_roi(stdout, stderr)

    assert isinstance(first, float)
    assert first == second


def test_evaluate_roi_monotonicity_stdout():
    stderr = "error" * 2
    low_stdout = "ok"
    high_stdout = "ok" * 50

    low_score = evaluate_roi(low_stdout, stderr)
    high_score = evaluate_roi(high_stdout, stderr)

    assert isinstance(low_score, float)
    assert isinstance(high_score, float)
    assert high_score >= low_score


def test_evaluate_roi_monotonicity_stderr():
    stdout = "result" * 5
    low_stderr = ""
    high_stderr = "fail" * 20

    low_score = evaluate_roi(stdout, low_stderr)
    high_score = evaluate_roi(stdout, high_stderr)

    assert isinstance(low_score, float)
    assert isinstance(high_score, float)
    assert high_score <= low_score


def test_evaluate_roi_edge_cases_json_serializable():
    empty_score = evaluate_roi("", "")
    stderr_only_score = evaluate_roi("", "error")
    large_stdout_score = evaluate_roi("x" * 100000, "")
    large_stderr_score = evaluate_roi("", "y" * 100000)
    non_ascii_score = evaluate_roi("\ufffd\ufffd", "\ufffd")

    for score in [
        empty_score,
        stderr_only_score,
        large_stdout_score,
        large_stderr_score,
        non_ascii_score,
    ]:
        assert isinstance(score, float)
        json.dumps(score)

    assert stderr_only_score <= empty_score
    assert large_stdout_score >= empty_score
    assert large_stderr_score <= empty_score
