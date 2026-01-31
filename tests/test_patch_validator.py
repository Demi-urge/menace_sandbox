import pytest

from patch_validator import validate_patch


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
                "id": "sig-empty-patched",
                "params": {"functions": ["greet"]},
            },
        ],
    )

    assert result["status"] == "failed"
    error = _find_error(result, "function signature missing")
    assert error["type"] == "PatchRuleError"
    mismatch = error["details"]["mismatch"]
    assert mismatch["symbol"] == "greet"
    assert mismatch["original"] is True
    assert mismatch["patched"] is False


def test_validate_patch_invalid_syntax_reports_patch_error() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        "def broken(:\n    pass",
        [
            {
                "type": "syntax_compile",
                "id": "syntax-check",
                "params": {"sources": ["patched"]},
            },
        ],
    )

    assert result["status"] == "failed"
    syntax_error = next(err for err in result["errors"] if err["type"] == "PatchSyntaxError")
    assert syntax_error["details"]["source"] == "patched"
    rule_error = _find_error(result, "syntax error")
    assert rule_error["details"]["code"] == "syntax_error"


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
                "id": "sig-missing",
                "params": {"functions": ["greet"]},
            }
        ],
    )

    assert result["status"] == "failed"
    mismatch = _find_error(result, "function signature missing")["details"]["mismatch"]
    assert mismatch["error"] == "missing_symbol"
    assert mismatch["original"] is True
    assert mismatch["patched"] is False


def test_validate_patch_unchanged_code_fails_rule_that_demands_changes() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        ORIGINAL_FUNCTION,
        [
            {
                "type": "signature_match",
                "id": "change-required",
                "params": {"functions": ["greet"], "require_changes": True},
            },
        ],
    )

    assert result["status"] == "failed"
    error = _find_error(result, "required change not detected")
    assert error["details"]["code"] == "unchanged_target"
    assert error["details"]["target"]["symbol"] == "greet"


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
                "id": "sig-mismatch",
                "params": {"functions": ["greet"]},
            },
        ],
    )

    assert result["status"] == "failed"
    mismatch = _find_error(result, "function signature mismatch")["details"]["mismatch"]
    assert mismatch["symbol"] == "greet"
    assert "original" in mismatch
    assert "patched" in mismatch


def test_validate_patch_rejects_malformed_rules_deterministically() -> None:
    result = validate_patch(
        ORIGINAL_FUNCTION,
        ORIGINAL_FUNCTION,
        [
            {"id": "missing-type"},
            {"type": "signature_match", "id": "missing-params"},
            {"type": 123, "id": "bad-type", "params": {}},
        ],
    )

    assert result["status"] == "failed"
    messages = {error["message"] for error in result["errors"] if error["type"] == "PatchValidationError"}
    assert "rule missing required field" in messages
    assert "rule type must be a string" in messages


@pytest.mark.parametrize(
    "rules",
    [
        pytest.param(
            [{"id": "missing-imports", "type": "required_imports", "params": {"imports": "oops"}}],
            id="missing-imports",
        ),
        pytest.param(
            [{"id": "missing-forbidden-constraints", "type": "forbidden_patterns", "params": {}}],
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

    assert result["status"] == "failed"
    assert any(error["type"] == "PatchValidationError" for error in result["errors"])
