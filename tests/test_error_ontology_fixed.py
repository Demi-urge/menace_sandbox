import pytest

from error_ontology import ErrorCategory, classify_error


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

    assert result["status"] == "ok"
    assert result["data"]["category"] == expected


def test_empty_input_returns_other():
    result = classify_error("")

    assert result["status"] == "fallback"
    assert result["errors"] == []
    assert result["data"]["category"] == ErrorCategory.Other.value


def test_identical_inputs_are_deterministic():
    payload = ["Syntax error in module", "syntax error in parser"]

    first = classify_error(payload)
    second = classify_error(payload)

    assert first["data"]["category"] == second["data"]["category"]
    assert first["meta"] == second["meta"]


def test_bundle_ambiguity_defaults_to_other():
    payload = ["missing config", "syntax error"]
    reversed_payload = list(reversed(payload))

    result = classify_error(payload)
    reversed_result = classify_error(reversed_payload)

    assert result["status"] == "fallback"
    assert reversed_result["status"] == "fallback"
    assert result["data"]["category"] == ErrorCategory.Other.value
    assert reversed_result["data"]["category"] == ErrorCategory.Other.value


def test_unknown_inputs_map_to_other():
    exception_result = classify_error(RuntimeError("boom"))
    text_result = classify_error("this does not match any phrase")

    assert exception_result["status"] == "fallback"
    assert text_result["status"] == "fallback"
    assert exception_result["data"]["category"] == ErrorCategory.Other.value
    assert text_result["data"]["category"] == ErrorCategory.Other.value
