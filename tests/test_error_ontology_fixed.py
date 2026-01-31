import pytest

from error_ontology import ErrorCategory, classify_error, _TOKEN_RULES


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Syntax error near token", ErrorCategory.SyntaxError.value),
        (ImportError("missing"), ErrorCategory.ImportError.value),
        (TypeError("wrong type"), ErrorCategory.TypeErrorMismatch.value),
        (AssertionError("contract violation"), ErrorCategory.ContractViolation.value),
        ("Edge case triggered", ErrorCategory.EdgeCaseFailure.value),
        ("Unhandled exception occurred", ErrorCategory.UnhandledException.value),
        (ValueError("invalid input"), ErrorCategory.InvalidInput.value),
        ("Missing return statement", ErrorCategory.MissingReturn.value),
        ("Config error detected", ErrorCategory.ConfigError.value),
    ],
)
def test_fixed_taxonomy_reachable(payload, expected):
    result = classify_error(payload)

    assert result["status"] == expected
    assert result["data"]["matched_rule_id"]


def test_empty_input_returns_other():
    result = classify_error("")

    assert result["status"] == ErrorCategory.Other.value
    assert result["errors"] == []
    assert result["data"]["matched_rule_id"] == "empty:other"


def test_identical_inputs_are_deterministic():
    payload = ["Syntax error in module", "syntax error in parser"]

    first = classify_error(payload)
    second = classify_error(payload)

    assert first["status"] == second["status"]
    assert first["meta"] == second["meta"]


def test_bundle_ambiguity_defaults_to_other():
    payload = ["missing config", "syntax error"]
    reversed_payload = list(reversed(payload))

    result = classify_error(payload)
    reversed_result = classify_error(reversed_payload)

    assert result["status"] == ErrorCategory.Other.value
    assert reversed_result["status"] == ErrorCategory.Other.value
    assert [item["status"] for item in result["data"]["bundle"]] == [
        ErrorCategory.ConfigError.value,
        ErrorCategory.SyntaxError.value,
    ]
    assert [item["status"] for item in reversed_result["data"]["bundle"]] == [
        ErrorCategory.SyntaxError.value,
        ErrorCategory.ConfigError.value,
    ]


def test_unknown_inputs_map_to_other():
    exception_result = classify_error(RuntimeError("boom"))
    text_result = classify_error("this does not match any phrase")

    assert exception_result["status"] == ErrorCategory.Other.value
    assert text_result["status"] == ErrorCategory.Other.value
    assert exception_result["data"]["matched_rule_id"] == "exception:unmatched"
    assert text_result["data"]["matched_rule_id"] == "text:unmatched"


def test_literal_phrase_matches_each_fixed_token_rule():
    for rule in _TOKEN_RULES:
        token = rule["match"]
        result = classify_error(f"before {token} after")
        assert result["status"] == rule["category"].value
        assert result["data"]["matched_token"] == token
