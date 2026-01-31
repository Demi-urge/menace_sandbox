from patch_validator import validate_patch


ORIGINAL_CODE = """
def greet(name):
    return f"Hello, {name}"
"""


def _assert_payload_shape(result: dict[str, object]) -> None:
    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert isinstance(result["data"], dict)
    assert isinstance(result["meta"], dict)
    assert isinstance(result["errors"], list)
    for error in result["errors"]:
        assert isinstance(error, dict)
        assert "type" in error
        assert error["type"] == error.get("error_type")
        assert "message" in error
        assert "details" in error


def _find_error_by_code(errors: list[dict[str, object]], code: str) -> dict[str, object]:
    for error in errors:
        details = error.get("details") or {}
        if details.get("code") == code:
            return error
    raise AssertionError(f"Expected error code {code} in {errors}")


def test_validate_patch_empty_patched_code_reports_missing_function() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "",
        [
            {
                "type": "mandatory_returns",
                "id": "must-return",
                "params": {"functions": ["greet"]},
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    error = _find_error_by_code(result["errors"], "mandatory_return_missing")
    failure = error["details"]["failure"]
    assert failure["function"] == "greet"
    assert failure["error"] == "missing_function"


def test_validate_patch_invalid_syntax_emits_syntax_error_entries() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "def broken(:\n    pass",
        [
            {
                "type": "syntax_compile",
                "id": "syntax-check",
                "params": {"sources": ["patched"]},
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    syntax_error = next(error for error in result["errors"] if error["type"] == "PatchSyntaxError")
    assert syntax_error["details"]["source"] == "patched"
    _find_error_by_code(result["errors"], "syntax_error")


def test_validate_patch_missing_required_function_flags_static_contract_failure() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        "def other():\n    return 1\n",
        [
            {
                "type": "static_contracts",
                "id": "contract-check",
                "params": {"functions": ["greet"], "require_docstring": True},
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    error = _find_error_by_code(result["errors"], "static_contract_missing")
    failure = error["details"]["failure"]
    assert failure["function"] == "greet"
    assert failure["error"] == "missing_function"


def test_validate_patch_unchanged_code_requires_removing_forbidden_pattern() -> None:
    code_with_eval = """
def greet(name):
    return eval(name)
"""
    result = validate_patch(
        code_with_eval,
        code_with_eval,
        [
            {
                "type": "forbidden_patterns",
                "id": "no-eval",
                "params": {"call_names": ["eval"]},
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    error = _find_error_by_code(result["errors"], "forbidden_pattern")
    match = error["details"]["match"]
    assert match["call"] == "eval"


def test_validate_patch_signature_mismatch_surfaces_details() -> None:
    patched = """
def greet(name, title=None):
    return f"Hello, {name}"
"""
    result = validate_patch(
        ORIGINAL_CODE,
        patched,
        [
            {
                "type": "signature_match",
                "id": "sig-match",
                "params": {"functions": ["greet"]},
            }
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    error = _find_error_by_code(result["errors"], "signature_mismatch")
    mismatch = error["details"]["mismatch"]
    assert mismatch["symbol"] == "greet"
    assert mismatch["kind"] == "function"


def test_validate_patch_rejects_malformed_rule_schema_fields() -> None:
    result = validate_patch(
        ORIGINAL_CODE,
        ORIGINAL_CODE,
        [
            {"id": "missing-params", "type": "signature_match"},
            {"id": "bad-type", "type": "unknown", "params": {}},
            {"id": "", "type": "syntax_compile", "params": {}},
        ],
    )

    _assert_payload_shape(result)

    assert result["status"] == "failed"
    validation_errors = [error for error in result["errors"] if error["type"] == "PatchValidationError"]
    assert validation_errors

    messages = {error["message"] for error in validation_errors}
    assert "rule missing required field" in messages
    assert "unsupported rule type" in messages
    assert "rule id must be a non-empty string" in messages

    fields = {
        error["details"].get("field") for error in validation_errors if "field" in error["details"]
    }
    assert "params" in fields
