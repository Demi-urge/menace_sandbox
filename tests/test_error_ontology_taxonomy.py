import pytest

from menace.error_ontology import ErrorCategory, classify_error


@pytest.mark.parametrize(
    "raw, expected_status, expected_token",
    [
        ("syntax error: invalid syntax", ErrorCategory.SyntaxError.value, "syntax error"),
        ("import error: no module named x", ErrorCategory.ImportError.value, "import error"),
        ("type mismatch: unsupported operand", ErrorCategory.TypeErrorMismatch.value, "type mismatch"),
        (
            "contract violation: invariant failed",
            ErrorCategory.ContractViolation.value,
            "contract violation",
        ),
        ("edge case triggered", ErrorCategory.EdgeCaseFailure.value, "edge case"),
        (
            "unhandled exception: boom",
            ErrorCategory.UnhandledException.value,
            "unhandled exception",
        ),
        ("invalid input: bad payload", ErrorCategory.InvalidInput.value, "invalid input"),
        ("missing return: handler", ErrorCategory.MissingReturn.value, "missing return"),
        ("config error: missing config", ErrorCategory.ConfigError.value, "config error"),
        ("Other: unspecified", ErrorCategory.Other.value, "other"),
    ],
)
def test_literal_tokens_map_to_expected_categories(raw, expected_status, expected_token):
    result = classify_error(raw)

    assert result["status"] == expected_status
    assert result["data"]["matched_token"] == expected_token
    if expected_token is not None:
        assert result["data"]["matched_rule_id"] == f"token:{expected_token}"


@pytest.mark.parametrize(
    "exc, expected_status",
    [
        (SyntaxError("bad"), ErrorCategory.SyntaxError.value),
        (ImportError("missing"), ErrorCategory.ImportError.value),
        (TypeError("bad"), ErrorCategory.TypeErrorMismatch.value),
        (AssertionError("contract"), ErrorCategory.ContractViolation.value),
        (ValueError("invalid"), ErrorCategory.InvalidInput.value),
        (KeyError("missing"), ErrorCategory.EdgeCaseFailure.value),
    ],
)
def test_exception_instances_map_to_expected_categories(exc, expected_status):
    result = classify_error(exc)

    assert result["status"] == expected_status
    assert result["data"]["matched_rule_id"].startswith("exception:")


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_empty_inputs_return_other(raw):
    result = classify_error(raw)

    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["matched_rule_id"] == "empty:other"


def test_partial_traceback_without_traceback_token_classifies_by_literal_tokens():
    raw = "File 'x.py', line 1: type mismatch: unsupported operand"
    result = classify_error(raw)

    assert result["status"] == ErrorCategory.TypeErrorMismatch.value
    assert result["data"]["matched_token"] == "type mismatch"


def test_multi_error_bundle_returns_other_and_aggregates_data():
    raw = ["invalid input: bad", "syntax error: nope", "Other: fallback"]
    result = classify_error(raw)

    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["bundle"] is not None
    assert [item["status"] for item in result["data"]["bundle"]] == [
        ErrorCategory.InvalidInput.value,
        ErrorCategory.SyntaxError.value,
        ErrorCategory.Other.value,
    ]
    assert result["meta"]["bundle_selected_index"] is None
    assert result["meta"]["bundle_rule"] == "always_other_for_bundle"


def test_multi_error_bundle_tuple_has_deterministic_status():
    raw = (ValueError("bad"), "UnhandledException: boom")
    result = classify_error(raw)

    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["bundle"] is not None
    assert len(result["data"]["bundle"]) == 2


def test_mapping_uses_deterministic_key_order():
    payload = {
        "error": "TypeError: bad",
        "traceback": "syntax error near token",
        "message": "unhandled exception: boom",
    }
    result = classify_error(payload)

    assert result["status"] == ErrorCategory.SyntaxError.value
    assert result["data"]["matched_rule_id"] == "token:syntax error"


def test_identical_inputs_produce_identical_outputs():
    raw = "InvalidInput: bad payload"

    assert classify_error(raw) == classify_error(raw)


def test_contextual_data_does_not_change_classification():
    raw = "type mismatch: cannot add"
    base = classify_error(raw)
    bundle = classify_error([raw, "Other: ignore"])
    mapping = classify_error({"error": raw, "errors": "Other: ignore"})

    assert base["status"] == ErrorCategory.TypeErrorMismatch.value
    assert bundle["data"]["bundle"][0]["status"] == base["status"]
    assert mapping["status"] == base["status"]
