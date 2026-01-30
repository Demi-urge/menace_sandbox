from menace.errors import PatchAnchorError, PatchRuleError, ValidationError
from menace_sandbox import patch_generator


def test_generate_patch_deterministic_output():
    source = "alpha\nbravo\ncharlie\ndelta\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-replace",
            "anchor": "alpha\n",
            "replacement": "alpha-1\n",
        },
        {
            "type": "delete_regex",
            "id": "rule-delete",
            "pattern": "charlie\\n",
        },
        {
            "type": "insert_after",
            "id": "rule-insert",
            "anchor": "delta\n",
            "content": "echo\n",
        },
    ]
    error_report = {}

    first = patch_generator.generate_patch(source, error_report, rules)
    second = patch_generator.generate_patch(source, error_report, rules)

    assert first == second
    assert list(first.keys()) == ["status", "data", "errors", "meta"]
    assert first["status"] == "ok"
    assert first["data"]["modified_source"] == "alpha-1\nbravo\ndelta\necho\n"
    assert first["data"]["patch_text"] == (
        "--- before\n"
        "+++ after\n"
        "@@ -1,5 +1,5 @@\n"
        "-alpha\n"
        "\n"
        "+alpha-1\n"
        "\n"
        " bravo\n"
        "\n"
        "-charlie\n"
        "\n"
        " delta\n"
        "\n"
        "+echo\n"
        "\n"
        " \n"
    )
    assert first["data"]["applied_rules"] == [
        {
            "id": "rule-replace",
            "type": "replace",
            "anchor": "alpha\n",
            "anchor_kind": "literal",
            "span": {"start": 0, "end": 6},
            "line_offsets": {"start_line": 1, "start_col": 0, "end_line": 2, "end_col": 0},
        },
        {
            "id": "rule-delete",
            "type": "delete_regex",
            "anchor": "charlie\\n",
            "anchor_kind": "regex",
            "span": {"start": 12, "end": 20},
            "line_offsets": {"start_line": 3, "start_col": 0, "end_line": 4, "end_col": 0},
        },
        {
            "id": "rule-insert",
            "type": "insert_after",
            "anchor": "delta\n",
            "anchor_kind": "literal",
            "span": {"start": 26, "end": 26},
            "line_offsets": {"start_line": 5, "start_col": 0, "end_line": 5, "end_col": 0},
        },
    ]
    assert first["meta"] == {
        "rule_count": 3,
        "changed_line_count": 4,
        "anchor_resolutions": [
            {"id": "rule-replace", "start": 0, "end": 6, "start_line": 1, "end_line": 2},
            {"id": "rule-delete", "start": 12, "end": 20, "start_line": 3, "end_line": 4},
            {"id": "rule-insert", "start": 26, "end": 26, "start_line": 5, "end_line": 5},
        ],
        "syntax_valid": True,
    }


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


def test_generate_patch_ambiguous_anchor_raises():
    source = "alpha\nalpha\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-1",
            "anchor": "alpha",
            "replacement": "alpha-1",
        }
    ]

    try:
        patch_generator.generate_patch(source, {}, rules)
    except PatchAnchorError as exc:
        assert exc.message == "anchor is ambiguous"
        assert exc.details == {"id": "rule-1", "anchor": "alpha", "match_count": 2}
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


def test_generate_patch_overlapping_delete_fails_without_partial_output():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-a",
            "anchor": "alpha\n",
            "replacement": "alpha-1\n",
        },
        {
            "type": "delete_regex",
            "id": "rule-b",
            "pattern": "alpha\\n",
        },
    ]

    result = patch_generator.generate_patch(source, {}, rules)

    assert result["status"] == "error"
    assert result["data"] == {"patch_text": "", "modified_source": "", "applied_rules": []}
    assert result["errors"] == [
        {
            "error_type": "PatchConflictError",
            "message": "conflicting edits detected",
            "details": {
                "rule_id": "rule-a",
                "conflicting_rule_id": "rule-b",
                "span": [0, 6],
                "conflicting_span": [0, 6],
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 2,
        "changed_line_count": 0,
        "anchor_resolutions": [
            {"id": "rule-a", "start": 0, "end": 6, "start_line": 1, "end_line": 2},
            {"id": "rule-b", "start": 0, "end": 6, "start_line": 1, "end_line": 2},
        ],
        "syntax_valid": None,
    }


def test_generate_patch_empty_rules_return_error():
    source = "alpha\nbravo\n"

    result = patch_generator.generate_patch(source, {}, [])

    assert result["status"] == "error"
    assert result["data"] == {"patch_text": "", "modified_source": "", "applied_rules": []}
    assert result["errors"] == [
        {
            "error_type": "ValidationError",
            "message": "No patch rules provided",
            "details": {"rule_count": 0},
        }
    ]
    assert result["meta"] == {
        "rule_count": 0,
        "changed_line_count": 0,
        "anchor_resolutions": [],
        "syntax_valid": None,
    }


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
    assert result["data"]["patch_text"] == (
        "--- before\n"
        "+++ after\n"
        "@@ -1,3 +1,4 @@\n"
        " def ok():\n"
        "\n"
        "+def bad(\n"
        "\n"
        "     return 1\n"
        "\n"
        " \n"
    )
    assert result["data"]["modified_source"] == "def ok():\ndef bad(\n    return 1\n"
    assert result["data"]["applied_rules"] == [
        {
            "id": "rule-1",
            "type": "insert_after",
            "anchor": "def ok():\n",
            "anchor_kind": "literal",
            "span": {"start": 10, "end": 10},
            "line_offsets": {"start_line": 2, "start_col": 0, "end_line": 2, "end_col": 0},
        }
    ]
    assert result["errors"] == [
        {
            "error_type": "ValidationError",
            "message": "Syntax check failed",
            "details": {
                "error_type": "SyntaxError",
                "message": "expected an indented block after function definition on line 1",
                "line": 2,
                "offset": 1,
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 1,
        "anchor_resolutions": [{"id": "rule-1", "start": 10, "end": 10, "start_line": 2, "end_line": 2}],
        "syntax_valid": False,
    }
