import pytest

from menace_sandbox.stabilization.patch_validator import validate_patch


ORIGINAL_CODE = """
def greet(name):
    return f"Hello, {name}"


def compute(x, y):
    return x + y
"""


def _assert_payload_shape(result: dict[str, object]) -> None:
    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert isinstance(result["errors"], list)
    for error in result["errors"]:
        assert isinstance(error, dict)
        assert "type" in error
        assert error["type"] == error.get("error_type")
        assert "message" in error
        assert "details" in error


def _assert_rule_context(error: dict[str, object], rule_index: int, rule_id: str | None = None) -> None:
    assert error.get("rule_index") == rule_index
    if rule_id is not None:
        assert error.get("rule_id") == rule_id
        details = error.get("details") or {}
        assert details.get("rule_id") == rule_id


def test_validate_patch_empty_patched_content_fails_with_clear_error() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "",
        [
            {
                "type": "signature_match",
                "target": "function",
                "match": "signature",
                "functions": ["greet"],
                "rule_id": "sig-missing",
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    assert result["errors"], "Expected errors for empty patched content"
    assert any(error["message"] == "function signature missing" for error in result["errors"])
    for error in result["errors"]:
        if error["message"] == "function signature missing":
            _assert_rule_context(error, rule_index=0)
            break


def test_validate_patch_reports_patch_syntax_error() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "def broken(:\n    pass",
        [],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    assert any(error["type"] == "PatchSyntaxError" for error in result["errors"])


def test_validate_patch_missing_required_function() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "def unrelated():\n    return 1\n",
        [
            {
                "type": "mandatory_returns",
                "target": "function",
                "match": "return",
                "functions": ["compute"],
                "rule_id": "mandatory-return",
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    assert any(error["message"] == "mandatory return requirement failed" for error in result["errors"])
    for error in result["errors"]:
        if error["message"] == "mandatory return requirement failed":
            _assert_rule_context(error, rule_index=0)
            break


def test_validate_patch_detects_static_contract_violation() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        ORIGINAL_CODE,
        [
            {
                "type": "static_contracts",
                "target": "function",
                "match": "docstring",
                "functions": ["compute"],
                "require_docstring": True,
                "rule_id": "missing-docstring",
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    assert any(error["message"] == "static contract violation" for error in result["errors"])
    for error in result["errors"]:
        if error["message"] == "static contract violation":
            _assert_rule_context(error, rule_index=0)
            break


def test_validate_patch_signature_mismatch() -> None:
    patched = """
def greet(name, title=None):
    return f"Hello, {name}"


def compute(x, y):
    return x + y
"""
    result = validate_patch(
        ORIGINAL_CODE,
        patched,
        [
            {
                "type": "signature_match",
                "target": "function",
                "match": "signature",
                "functions": ["greet"],
                "rule_id": "sig-change",
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    assert any(error["message"] == "function signature mismatch" for error in result["errors"])
    for error in result["errors"]:
        if error["message"] == "function signature mismatch":
            _assert_rule_context(error, rule_index=0)
            break


def test_validate_patch_malformed_rules_return_rule_error() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        ORIGINAL_CODE,
        [
            {
                "type": 123,
                "target": "module",
                "match": "hash",
                "rule_id": "bad-rule",
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    rule_errors = [error for error in result["errors"] if error["type"] == "PatchRuleError"]
    assert rule_errors
    assert any("rule type must be a string" == error["message"] for error in rule_errors)
    for error in rule_errors:
        if error["message"] == "rule type must be a string":
            _assert_rule_context(error, rule_index=0, rule_id="bad-rule")
            break


def test_validate_patch_reports_multiple_failures() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        ORIGINAL_CODE,
        [
            {
                "type": "required_imports",
                "target": "module",
                "match": "imports",
                "imports": [{"kind": "import", "module": "json"}],
            },
            {
                "type": "static_contracts",
                "target": "function",
                "match": "docstring",
                "functions": ["compute"],
                "require_docstring": True,
            },
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "fail"
    messages = {error["message"] for error in result["errors"]}
    assert "required import missing" in messages
    assert "static contract violation" in messages
    assert len(result["errors"]) >= 2
    for error in result["errors"]:
        if error["message"] == "required import missing":
            _assert_rule_context(error, rule_index=0)
        if error["message"] == "static contract violation":
            _assert_rule_context(error, rule_index=1)
