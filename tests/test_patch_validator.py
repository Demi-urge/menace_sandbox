import pytest

from menace_sandbox.stabilization.patch_code_validator import validate_patch


ORIGINAL_FUNCTION = """
def greet(name):
    return name
"""


def _find_error(result, message: str):
    return next(error for error in result["errors"] if error["message"] == message)


def test_validate_patch_empty_patched_source_fails_signature_match() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        "",
        [
            {
                "type": "signature_match",
                "functions": ["greet"],
                "rule_id": "sig-empty-patched",
            }
        ],
    )

    assert result["status"] == "fail"
    error = _find_error(result, "signature mismatch")
    assert error["type"] == "PatchRuleError"
    assert error["details"]["mismatches"] == [
        {
            "function": "greet",
            "error": "missing_function",
            "original": True,
            "patched": False,
        }
    ]


def test_validate_patch_invalid_syntax_reports_patch_error() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        "def broken(:\n    pass",
        [
            {
                "type": "syntax",
                "rule_id": "syntax-check",
            }
        ],
    )

    assert result["status"] == "fail"
    syntax_error = next(err for err in result["errors"] if err["type"] == "PatchSyntaxError")
    assert syntax_error["details"]["source"] == "patched"


def test_validate_patch_missing_function_required_by_rule() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        """
def other():
    return 1
""",
        [
            {
                "type": "signature_match",
                "functions": ["greet"],
                "rule_id": "sig-missing",
            }
        ],
    )

    assert result["status"] == "fail"
    mismatch = _find_error(result, "signature mismatch")["details"]["mismatches"][0]
    assert mismatch["error"] == "missing_function"
    assert mismatch["original"] is True
    assert mismatch["patched"] is False


def test_validate_patch_unchanged_code_fails_rule_that_demands_changes() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        ORIGINAL_FUNCTION,
        [
            {
                "type": "static_contracts",
                "functions": ["greet"],
                "require_docstring": True,
                "rule_id": "docstring-required",
            }
        ],
    )

    assert result["status"] == "fail"
    error = _find_error(result, "static contract failure")
    assert error["details"]["failures"] == [
        {"function": "greet", "error": "missing_docstring"}
    ]


def test_validate_patch_signature_mismatch_between_versions() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        """
def greet(name, title=None):
    return name
""",
        [
            {
                "type": "signature_match",
                "functions": ["greet"],
                "rule_id": "sig-mismatch",
            }
        ],
    )

    assert result["status"] == "fail"
    mismatch = _find_error(result, "signature mismatch")["details"]["mismatches"][0]
    assert mismatch["function"] == "greet"
    assert "original" in mismatch
    assert "patched" in mismatch


def test_validate_patch_rejects_malformed_rules_deterministically() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        ORIGINAL_FUNCTION,
        [
            {"rule_id": "missing-type"},
            {"type": "signature_match", "rule_id": "missing-functions"},
            {"type": 123, "rule_id": "bad-type"},
        ],
    )

    assert result["status"] == "fail"
    messages = {error["message"] for error in result["errors"] if error["type"] == "PatchRuleError"}
    assert "rule type must be a string" in messages
    assert "signature_match requires function names" in messages


@pytest.mark.parametrize(
    "rules",
    [
        pytest.param(
            [{"type": "required_imports", "imports": []}],
            id="missing-imports",
        ),
        pytest.param(
            [{"type": "forbidden_patterns"}],
            id="missing-forbidden-constraints",
        ),
    ],
)
def test_validate_patch_malformed_rules_payloads(rules) -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        ORIGINAL_FUNCTION,
        rules,
    )

    assert result["status"] == "fail"
    assert any(error["type"] == "PatchRuleError" for error in result["errors"])
