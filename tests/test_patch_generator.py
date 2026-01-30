import difflib

from menace_sandbox import patch_generator


def _build_expected_diff(before: str, after: str) -> str:
    before_lines = before.splitlines()
    if before.endswith("\n"):
        before_lines.append("")
    after_lines = after.splitlines()
    if after.endswith("\n"):
        after_lines.append("")
    diff_lines = difflib.unified_diff(
        [f"{line}\n" for line in before_lines],
        [f"{line}\n" for line in after_lines],
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(diff_lines)


def _assert_deterministic(source: str, error_report: dict, rules: list[dict]) -> dict:
    first = patch_generator.generate_patch(source, error_report, rules)
    second = patch_generator.generate_patch(source, error_report, rules)
    assert first == second
    return first


def test_empty_rules_noop_patch():
    source = "alpha\nbravo\n"

    result = _assert_deterministic(source, {}, [])

    assert result["status"] == "noop"
    assert result["data"] == {
        "patch_text": "",
        "modified_source": source,
        "applied_rules": [],
    }
    assert result["errors"] == []
    assert result["meta"] == {
        "rule_count": 0,
        "changed_line_count": 0,
        "anchor_resolutions": [],
        "syntax_valid": None,
        "original_source": source,
    }


def test_single_replace_rule():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-replace",
            "anchor": "alpha\n",
            "replacement": "alpha-1\n",
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha-1\nbravo\n"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["data"]["modified_source"] == expected_source
    assert result["data"]["patch_text"] == _build_expected_diff(source, expected_source)
    assert result["data"]["applied_rules"] == [
        {
            "id": "rule-replace",
            "type": "replace",
            "anchor": "alpha\n",
            "anchor_kind": "literal",
            "span": {"start": 0, "end": 6},
            "line_offsets": {
                "start_line": 1,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 2,
        "anchor_resolutions": [{"id": "rule-replace", "start": 0, "end": 6, "start_line": 1, "end_line": 2}],
        "syntax_valid": None,
        "original_source": source,
    }


def test_single_insert_after_rule():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-insert",
            "anchor": "alpha\n",
            "content": "beta\n",
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha\nbeta\nbravo\n"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["data"]["modified_source"] == expected_source
    assert result["data"]["patch_text"] == _build_expected_diff(source, expected_source)
    assert result["data"]["applied_rules"] == [
        {
            "id": "rule-insert",
            "type": "insert_after",
            "anchor": "alpha\n",
            "anchor_kind": "literal",
            "span": {"start": 6, "end": 6},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 1,
        "anchor_resolutions": [{"id": "rule-insert", "start": 6, "end": 6, "start_line": 2, "end_line": 2}],
        "syntax_valid": None,
        "original_source": source,
    }


def test_single_delete_regex_rule():
    source = "alpha\nbravo\ncharlie\n"
    rules = [
        {
            "type": "delete_regex",
            "id": "rule-delete",
            "pattern": "bravo",
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha\ncharlie\n"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["data"]["modified_source"] == expected_source
    assert result["data"]["patch_text"] == _build_expected_diff(source, expected_source)
    assert result["data"]["applied_rules"] == [
        {
            "id": "rule-delete",
            "type": "delete_regex",
            "anchor": "bravo",
            "anchor_kind": "regex",
            "span": {"start": 6, "end": 12},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 3,
                "end_col": 0,
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 1,
        "anchor_resolutions": [{"id": "rule-delete", "start": 6, "end": 12, "start_line": 2, "end_line": 3}],
        "syntax_valid": None,
        "original_source": source,
    }


def test_conflicting_edits_fail_deterministically():
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

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["data"] == {"patch_text": "", "modified_source": "", "applied_rules": []}
    assert result["errors"] == [
        {
            "error_type": "PatchConflictError",
            "message": "conflicting edits detected",
            "details": {
                "rule_id": "rule-a",
                "conflicting_rule_id": "rule-b",
                "span": [0, 5],
                "conflicting_span": [0, 5],
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 2,
        "changed_line_count": 0,
        "anchor_resolutions": [
            {"id": "rule-a", "start": 0, "end": 5, "start_line": 1, "end_line": 1},
            {"id": "rule-b", "start": 0, "end": 5, "start_line": 1, "end_line": 1},
        ],
        "syntax_valid": None,
        "original_source": source,
    }


def test_overlapping_ranges_fail_deterministically():
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
            "pattern": "alpha",
        },
    ]

    result = _assert_deterministic(source, {}, rules)

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
        "original_source": source,
    }


def test_ambiguous_anchor_fails_with_patch_anchor_error():
    source = "alpha\nalpha\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-1",
            "anchor": "alpha",
            "replacement": "alpha-1",
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["data"] == {"patch_text": "", "modified_source": "", "applied_rules": []}
    assert result["errors"] == [
        {
            "error_type": "PatchAnchorError",
            "message": "anchor is ambiguous",
            "details": {
                "rule_index": 0,
                "rule": {
                    "rule_id": "rule-1",
                    "anchor": "alpha",
                    "replacement": "alpha-1",
                    "anchor_kind": "literal",
                    "meta": None,
                },
                "rule_id": "rule-1",
                "anchor_search": {
                    "anchor": "alpha",
                    "anchor_kind": "literal",
                    "match_count": 2,
                    "matches": [(0, 5), (6, 11)],
                },
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 0,
        "anchor_resolutions": [],
        "syntax_valid": None,
        "original_source": source,
    }


def test_syntax_breaking_insert_fails_with_patch_syntax_error():
    source = "def ok():\n    return 1\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-1",
            "anchor": "def ok():\n",
            "content": "def bad(\n",
        }
    ]

    result = _assert_deterministic(source, {"file_path": "example.py"}, rules)

    expected_source = "def ok():\ndef bad(\n    return 1\n"
    assert result["status"] == "error"
    assert result["data"]["modified_source"] == expected_source
    assert result["data"]["patch_text"] == _build_expected_diff(source, expected_source)
    assert result["data"]["applied_rules"] == [
        {
            "id": "rule-1",
            "type": "insert_after",
            "anchor": "def ok():\n",
            "anchor_kind": "literal",
            "span": {"start": 10, "end": 10},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
    assert result["errors"] == [
        {
            "error_type": "PatchSyntaxError",
            "message": "Syntax check failed",
            "details": {
                "error_type": "SyntaxError",
                "message": "expected an indented block after function definition on line 1",
                "line": 2,
                "offset": 1,
                "rule_ids": ["rule-1"],
                "rule_count": 1,
            },
        }
    ]
    assert result["meta"] == {
        "rule_count": 1,
        "changed_line_count": 1,
        "anchor_resolutions": [{"id": "rule-1", "start": 10, "end": 10, "start_line": 2, "end_line": 2}],
        "syntax_valid": False,
        "original_source": source,
    }
