import pytest

from menace.error_ontology import ErrorCategory, classify_error, _TOKEN_RULES


@pytest.mark.parametrize("raw", [
    "TypeError: type mismatch",
    {"error": "type mismatch", "detail": "bad"},
])
def test_classify_error_returns_required_keys_and_types(raw):
    result = classify_error(raw)

    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert isinstance(result["status"], str)
    assert isinstance(result["data"], dict)
    assert isinstance(result["errors"], list)
    assert isinstance(result["meta"], dict)

    data = result["data"]
    assert set(data.keys()) == {
        "input_kind",
        "normalized",
        "matched_token",
        "matched_rule_id",
        "bundle",
    }
    assert isinstance(data["input_kind"], str)
    assert isinstance(data["normalized"], str)
    assert data["matched_token"] is None or isinstance(data["matched_token"], str)
    assert isinstance(data["matched_rule_id"], str)


def test_identical_inputs_yield_identical_categories():
    log = "missing return in handler"
    categories = [classify_error(log)["status"] for _ in range(4)]
    assert categories == [categories[0]] * 4


def test_unrelated_context_does_not_change_literal_match():
    base = "type mismatch: cannot add"
    augmented = f"{base} | extra context unrelated to error"
    assert (
        classify_error(base)["status"]
        == classify_error(augmented)["status"]
        == ErrorCategory.TypeErrorMismatch.value
    )


@pytest.mark.parametrize("raw", [
    None,
    "",
    "   ",
    "Traceback (most recent call last):\n  File \"x.py\", line 1",
])
def test_empty_or_partial_tracebacks_return_other(raw):
    assert classify_error(raw)["status"] == ErrorCategory.Other.value


def test_multi_error_bundle_and_unknown_exception_return_other():
    class UnknownError(Exception):
        pass

    inputs = ["Traceback (most recent call last):", UnknownError("boom")]
    result = classify_error(inputs)
    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["bundle"] is not None
    assert [item["status"] for item in result["data"]["bundle"]] == [
        ErrorCategory.Other.value,
        ErrorCategory.Other.value,
    ]


def test_mapping_with_sequence_normalizes_to_bundle():
    payload = {"errors": ["type mismatch in handler", "Other: fallback"]}
    result = classify_error(payload)

    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["bundle"] is not None
    assert [item["status"] for item in result["data"]["bundle"]] == [
        ErrorCategory.TypeErrorMismatch.value,
        ErrorCategory.Other.value,
    ]


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
    "config error: missing config",
])
def test_classify_error_returns_only_fixed_taxonomy_values(raw):
    allowed = {category.value for category in ErrorCategory}
    status = classify_error(raw)["status"]
    assert status in allowed


def test_literal_phrase_matches_fixed_token_list():
    for rule in _TOKEN_RULES:
        token = rule["match"]
        result = classify_error(f"{token} in handler")
        assert result["status"] == rule["category"].value
        assert result["data"]["matched_token"] == token
