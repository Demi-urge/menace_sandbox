import pytest

from menace.error_ontology import ErrorCategory, classify_error


@pytest.mark.parametrize("raw", [
    "TypeError: unsupported operand",
    {"error": "type error", "detail": "bad"},
])
def test_classify_error_returns_required_keys_and_types(raw):
    result = classify_error(raw)

    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert isinstance(result["status"], str)
    assert isinstance(result["data"], dict)
    assert isinstance(result["errors"], list)
    assert isinstance(result["meta"], dict)

    data = result["data"]
    assert set(data.keys()) == {"category", "source", "matched_rule"}
    assert isinstance(data["category"], str)
    assert isinstance(data["source"], str)
    assert isinstance(data["matched_rule"], str)


def test_identical_inputs_yield_identical_categories():
    log = "missing return in handler"
    categories = [classify_error(log)["data"]["category"] for _ in range(4)]
    assert categories == [categories[0]] * 4


def test_unrelated_context_does_not_change_literal_match():
    base = "type error: cannot add"
    augmented = f"{base} | extra context unrelated to error"
    assert (
        classify_error(base)["data"]["category"]
        == classify_error(augmented)["data"]["category"]
        == ErrorCategory.TypeErrorMismatch.value
    )


@pytest.mark.parametrize("raw", [
    None,
    "",
    "   ",
    "Traceback (most recent call last):\n  File \"x.py\", line 1",
])
def test_empty_or_partial_tracebacks_return_other(raw):
    assert classify_error(raw)["data"]["category"] == ErrorCategory.Other.value


def test_multi_error_bundle_and_unknown_exception_return_other():
    class UnknownError(Exception):
        pass

    inputs = ["Traceback (most recent call last):", UnknownError("boom")]
    result = classify_error(inputs)
    assert result["data"]["category"] == ErrorCategory.Other.value


@pytest.mark.parametrize("raw", [
    None,
    "syntax error near token",
    "no module named requests",
    "type mismatch in handler",
    "contract violation detected",
    "edge case triggered",
    "unhandled exception: boom",
    "invalid input payload",
    "missing return in handler",
    "configuration error: missing config",
])
def test_classify_error_returns_only_fixed_taxonomy_values(raw):
    allowed = {category.value for category in ErrorCategory}
    category = classify_error(raw)["data"]["category"]
    assert category in allowed
