import pytest

from error_ontology import ErrorCategory, classify_error


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Syntax error near token", ErrorCategory.SyntaxError),
        (ImportError("missing"), ErrorCategory.ImportError),
        (TypeError("wrong type"), ErrorCategory.TypeErrorMismatch),
        (AssertionError("contract violation"), ErrorCategory.ContractViolation),
        ("Edge case triggered", ErrorCategory.EdgeCaseFailure),
        ("Unhandled exception occurred", ErrorCategory.UnhandledException),
        (ValueError("invalid input"), ErrorCategory.InvalidInput),
        ("Missing return statement", ErrorCategory.MissingReturn),
        ("Config error detected", ErrorCategory.ConfigError),
        ("totally unrecognized", ErrorCategory.Other),
    ],
)
def test_fixed_taxonomy_reachable(payload, expected):
    result = classify_error(payload)

    assert result["status"] == "ok"
    assert result["data"]["category"] == expected


def test_empty_input_returns_other():
    result = classify_error("")

    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["data"]["category"] == ErrorCategory.Other


def test_identical_inputs_are_deterministic():
    payload = ["Syntax error in module", "missing return"]

    first = classify_error(payload)
    second = classify_error(payload)

    assert first["data"]["category"] == second["data"]["category"]
    assert first["meta"] == second["meta"]


def test_priority_order_is_deterministic_for_lists():
    payload = ["missing config", "syntax error"]
    reversed_payload = list(reversed(payload))

    result = classify_error(payload)
    reversed_result = classify_error(reversed_payload)

    assert result["data"]["category"] == ErrorCategory.SyntaxError
    assert reversed_result["data"]["category"] == ErrorCategory.SyntaxError


def test_unknown_inputs_map_to_other():
    exception_result = classify_error(RuntimeError("boom"))
    text_result = classify_error("this does not match any phrase")

    assert exception_result["data"]["category"] == ErrorCategory.Other
    assert text_result["data"]["category"] == ErrorCategory.Other
