import pytest

from menace_sandbox.stabilization.patch_code_validator import validate_patch


MINIMAL_FUNCTION = """
def greet(name):
    return name
"""


def test_validate_patch_empty_source_reports_missing_function() -> None:
    result = validate_patch(
        "",
        "",
        [
            {
                "type": "signature_match",
                "functions": ["greet"],
                "rule_id": "sig-empty",
            }
        ],
    )

    assert result["status"] == "fail"
    error = next(err for err in result["errors"] if err["message"] == "signature mismatch")
    assert error["type"] == "PatchRuleError"
    mismatches = error["details"]["mismatches"]
    assert mismatches == [
        {
            "function": "greet",
            "error": "missing_function",
            "original": False,
            "patched": False,
        }
    ]


def test_validate_patch_surfaces_patch_syntax_error() -> None:
    result = validate_patch(
        MINIMAL_FUNCTION,
        "def broken(:\n    pass",
        [],
    )

    assert result["status"] == "fail"
    syntax_error = next(err for err in result["errors"] if err["type"] == "PatchSyntaxError")
    assert syntax_error["details"]["source"] == "patched"


def test_validate_patch_missing_required_function_reports_error() -> None:
    result = validate_patch(
        MINIMAL_FUNCTION,
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
    error = next(err for err in result["errors"] if err["message"] == "signature mismatch")
    mismatch = error["details"]["mismatches"][0]
    assert mismatch["error"] == "missing_function"
    assert mismatch["original"] is True
    assert mismatch["patched"] is False


def test_validate_patch_unchanged_code_hits_static_contract_failure() -> None:
    result = validate_patch(
        MINIMAL_FUNCTION,
        MINIMAL_FUNCTION,
        [
            {
                "type": "static_contracts",
                "functions": ["greet"],
                "require_docstring": True,
                "rule_id": "doc-required",
            }
        ],
    )

    assert result["status"] == "fail"
    error = next(err for err in result["errors"] if err["message"] == "static contract failure")
    assert error["type"] == "PatchRuleError"
    assert error["details"]["failures"] == [
        {"function": "greet", "error": "missing_docstring"}
    ]


def test_validate_patch_signature_mismatch_reports_details() -> None:
    result = validate_patch(
        MINIMAL_FUNCTION,
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
    error = next(err for err in result["errors"] if err["message"] == "signature mismatch")
    mismatches = error["details"]["mismatches"]
    assert mismatches[0]["function"] == "greet"
    assert "original" in mismatches[0]
    assert "patched" in mismatches[0]


def test_validate_patch_rejects_malformed_rules() -> None:
    result = validate_patch(
        MINIMAL_FUNCTION,
        MINIMAL_FUNCTION,
        [
            {"type": "unknown_rule", "rule_id": "bad-type"},
            {"type": "signature_match", "rule_id": "missing-fns"},
            {"type": "signature_match", "functions": ["greet"], "require_annotations": "no"},
        ],
    )

    assert result["status"] == "fail"
    messages = {error["message"] for error in result["errors"] if error["type"] == "PatchRuleError"}
    assert "unsupported rule type" in messages
    assert "signature_match requires function names" in messages
    assert "require_annotations must be a bool" in messages
