from menace.errors import PatchAnchorError, PatchRuleError, ValidationError
from menace_sandbox import patch_generator


def test_generate_patch_deterministic_output():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-1",
            "anchor": "alpha",
            "replacement": "alpha-1",
        },
        {
            "type": "insert_after",
            "id": "rule-2",
            "anchor": "bravo\n",
            "content": "charlie\n",
        },
    ]
    error_report = {"file_path": "module.py"}

    first = patch_generator.generate_patch(source, error_report, rules)
    second = patch_generator.generate_patch(source, error_report, rules)

    assert first == second
    assert list(first.keys()) == ["status", "data", "errors", "meta"]
    assert first["status"] == "ok"
    assert first["data"]["modified_source"] == "alpha-1\nbravo\ncharlie\n"
    assert first["data"]["applied_rules"][0]["id"] == "rule-1"


def test_generate_patch_validation_errors_raise():
    source = "alpha\nbravo\n"
    error_report = {}

    try:
        patch_generator.generate_patch(source, error_report, [{"type": "unknown", "id": "x"}])
    except PatchRuleError as exc:
        assert exc.message == "Unknown rule type"
    else:
        raise AssertionError("Expected PatchRuleError")


def test_generate_patch_missing_anchor_raises():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-1",
            "anchor": "charlie",
            "content": "delta",
        }
    ]

    try:
        patch_generator.generate_patch(source, {}, rules)
    except PatchAnchorError as exc:
        assert exc.message == "anchor not found"
    else:
        raise AssertionError("Expected PatchAnchorError")


def test_generate_patch_conflicting_replacements_fail():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-a",
            "anchor": "alpha",
            "replacement": "alpha-1",
        },
        {
            "type": "replace",
            "id": "rule-b",
            "anchor": "alpha",
            "replacement": "alpha-2",
        },
    ]

    result = patch_generator.generate_patch(source, {}, rules)

    assert result["status"] == "error"
    error = result["errors"][0]
    assert error["error_type"] == "PatchConflictError"
    assert error["message"] == "conflicting edits detected"
    assert error["details"]["conflicting_rule_id"] == "rule-b"


def test_generate_patch_empty_rules_noop():
    source = "alpha\nbravo\n"

    result = patch_generator.generate_patch(source, {}, [])

    assert result["status"] == "ok"
    assert result["data"]["patch_text"] == ""
    assert result["data"]["modified_source"] == source
    assert result["errors"] == []


def test_generate_patch_syntax_error_reported():
    source = "def ok():\n    return 1\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-1",
            "anchor": "def ok():\n",
            "content": "def bad(\n",
        }
    ]

    result = patch_generator.generate_patch(source, {"file_path": "example.py"}, rules)

    assert result["status"] == "error"
    assert result["errors"]
    error = result["errors"][0]
    assert error["error_type"] == "ValidationError"
    assert error["message"] == "Syntax check failed"
    assert error["details"]["error_type"] == "SyntaxError"
