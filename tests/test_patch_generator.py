import difflib
import hashlib
import json

import pytest

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


def _input_hash(source: str, rule_summaries: list[dict]) -> str:
    payload = {"source": source, "rules": rule_summaries}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_meta(
    *,
    source: str,
    rule_summaries: list[dict],
    applied_rules: list[dict],
    anchor_resolutions: list[dict],
    changed_line_count: int,
    syntax_valid: bool | None,
    notes: list[str] | None = None,
) -> dict:
    applied_rule_ids = [rule["id"] for rule in applied_rules]
    return {
        "rule_summaries": rule_summaries,
        "rule_count": len(rule_summaries),
        "applied_count": len(applied_rules),
        "total_changes": len(applied_rules),
        "applied_rule_ids": applied_rule_ids,
        "rule_application_order": applied_rule_ids,
        "applied_rules": applied_rules,
        "anchor_resolutions": anchor_resolutions,
        "changed_line_count": changed_line_count,
        "syntax_valid": syntax_valid,
        "validation_results": {"syntax_valid": syntax_valid},
        "input_hash": _input_hash(source, rule_summaries),
        "notes": list(notes or []),
    }


def _assert_deterministic(source: str, error_report: dict, rules: list[dict]) -> dict:
    first = patch_generator.generate_patch(source, error_report, rules)
    second = patch_generator.generate_patch(source, error_report, rules)
    assert first == second
    return first


def test_empty_rules_return_deterministic_error_payload():
    source = "alpha\nbravo\n"

    result = _assert_deterministic(source, {}, [])

    assert result["status"] == "error"
    assert result["data"] == {
        "patch_text": "",
        "modified_source": "",
        "applied_rules": [],
        "changes": [],
        "audit_trail": [],
    }
    assert result["errors"][0]["type"] == "PatchRuleError"
    assert result["errors"][0]["rule_id"] is None
    assert result["errors"][0]["rule_index"] is None
    assert result["errors"][0]["details"]["field"] == "rules"
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=[],
        applied_rules=[],
        anchor_resolutions=[],
        changed_line_count=0,
        syntax_valid=None,
        notes=["empty_rules"],
    )


def test_single_replace_rule():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-replace",
            "description": "replace alpha",
            "anchor": "alpha\n",
            "replacement": "alpha-1\n",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha-1\nbravo\n"
    rule_summaries = [{"id": "rule-replace", "type": "replace", "description": "replace alpha"}]
    applied_rules = [
        {
            "id": "rule-replace",
            "type": "replace",
            "rationale": "replace alpha",
            "span": {"start": 0, "end": 6},
            "line_offsets": {
                "start_line": 1,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
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
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=rule_summaries,
        applied_rules=applied_rules,
        anchor_resolutions=[{"id": "rule-replace", "start": 0, "end": 6, "start_line": 1, "end_line": 2}],
        changed_line_count=2,
        syntax_valid=None,
    )


def test_single_insert_after_rule():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-insert",
            "description": "insert beta",
            "anchor": "alpha\n",
            "content": "beta\n",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha\nbeta\nbravo\n"
    rule_summaries = [{"id": "rule-insert", "type": "insert_after", "description": "insert beta"}]
    applied_rules = [
        {
            "id": "rule-insert",
            "type": "insert_after",
            "rationale": "insert beta",
            "span": {"start": 6, "end": 6},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
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
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=rule_summaries,
        applied_rules=applied_rules,
        anchor_resolutions=[{"id": "rule-insert", "start": 6, "end": 6, "start_line": 2, "end_line": 2}],
        changed_line_count=1,
        syntax_valid=None,
    )


def test_single_delete_regex_rule():
    source = "alpha\nbravo\ncharlie\n"
    rules = [
        {
            "type": "delete_regex",
            "id": "rule-delete",
            "description": "delete bravo",
            "pattern": "bravo",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha\ncharlie\n"
    rule_summaries = [{"id": "rule-delete", "type": "delete_regex", "description": "delete bravo"}]
    applied_rules = [
        {
            "id": "rule-delete",
            "type": "delete_regex",
            "rationale": "delete bravo",
            "span": {"start": 6, "end": 12},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 3,
                "end_col": 0,
            },
        }
    ]
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
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=rule_summaries,
        applied_rules=applied_rules,
        anchor_resolutions=[{"id": "rule-delete", "start": 6, "end": 12, "start_line": 2, "end_line": 3}],
        changed_line_count=1,
        syntax_valid=None,
    )


def test_replace_insert_delete_rules_apply_deterministically():
    source = "alpha\nbravo\ncharlie\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-replace",
            "description": "replace bravo",
            "anchor": "bravo\n",
            "replacement": "bravo-1\n",
            "meta": {"source": "test"},
        },
        {
            "type": "insert_after",
            "id": "rule-insert",
            "description": "insert beta",
            "anchor": "alpha\n",
            "content": "beta\n",
            "meta": {"source": "test"},
        },
        {
            "type": "delete_regex",
            "id": "rule-delete",
            "description": "delete charlie",
            "pattern": "charlie",
            "meta": {"source": "test"},
        },
    ]

    result = _assert_deterministic(source, {}, rules)

    expected_source = "alpha\nbeta\nbravo-1\n"
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["data"]["modified_source"] == expected_source
    assert result["data"]["patch_text"] == _build_expected_diff(source, expected_source)
    assert [rule["id"] for rule in result["data"]["applied_rules"]] == [
        "rule-insert",
        "rule-replace",
        "rule-delete",
    ]


def test_missing_anchor_fails_with_patch_anchor_error():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-missing",
            "description": "missing anchor",
            "anchor": "missing\n",
            "content": "beta\n",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["type"] == "PatchAnchorError"
    assert result["errors"][0]["details"]["anchor_search"]["match_count"] == 0
    assert result["errors"][0]["details"]["anchor_search"]["anchor"] == "missing\n"


def test_invalid_regex_pattern_fails_with_patch_rule_error():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "delete_regex",
            "id": "rule-invalid",
            "description": "invalid regex",
            "pattern": "(unclosed",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["type"] == "PatchRuleError"
    assert result["errors"][0]["message"] == "delete_regex pattern is invalid"


def test_multiple_inserts_same_anchor_fails_deterministically():
    source = "alpha\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-insert-a",
            "description": "insert beta",
            "anchor": "alpha\n",
            "content": "beta\n",
            "meta": {"source": "test"},
        },
        {
            "type": "insert_after",
            "id": "rule-insert-b",
            "description": "insert gamma",
            "anchor": "alpha\n",
            "content": "gamma\n",
            "meta": {"source": "test"},
        },
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["type"] == "PatchConflictError"
    assert result["errors"][0]["message"] == "multiple inserts at the same anchor are not allowed"


def test_conflicting_edits_fail_deterministically():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-a",
            "description": "replace alpha once",
            "anchor": "alpha",
            "replacement": "alpha-1",
            "meta": {"source": "test"},
        },
        {
            "type": "replace",
            "id": "rule-b",
            "description": "replace alpha twice",
            "anchor": "alpha",
            "replacement": "alpha-2",
            "meta": {"source": "test"},
        },
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["data"] == {
        "patch_text": "",
        "modified_source": "",
        "applied_rules": [],
        "changes": [],
        "audit_trail": [],
    }
    assert result["errors"] == [
        {
            "type": "PatchConflictError",
            "error_type": "PatchConflictError",
            "message": "conflicting edits detected",
            "rule_id": "rule-a",
            "rule_index": 0,
            "details": {
                "rule_id": "rule-a",
                "rule_index": 0,
                "conflicting_rule_id": "rule-b",
                "conflicting_rule_index": 1,
                "rule_type": "replace",
                "conflicting_rule_type": "replace",
                "span": [0, 5],
                "conflicting_span": [0, 5],
                "line_offsets": {
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 1,
                    "end_col": 5,
                },
                "conflicting_line_offsets": {
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 1,
                    "end_col": 5,
                },
            },
            "location": {
                "start_line": 1,
                "start_col": 0,
                "end_line": 1,
                "end_col": 5,
            },
        }
    ]
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=[
            {"id": "rule-a", "type": "replace", "description": "replace alpha once"},
            {"id": "rule-b", "type": "replace", "description": "replace alpha twice"},
        ],
        applied_rules=[],
        anchor_resolutions=[
            {"id": "rule-a", "start": 0, "end": 5, "start_line": 1, "end_line": 1},
            {"id": "rule-b", "start": 0, "end": 5, "start_line": 1, "end_line": 1},
        ],
        changed_line_count=0,
        syntax_valid=None,
    )


def test_overlapping_ranges_fail_deterministically():
    source = "alpha\nbravo\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-a",
            "description": "replace alpha",
            "anchor": "alpha\n",
            "replacement": "alpha-1\n",
            "meta": {"source": "test"},
        },
        {
            "type": "delete_regex",
            "id": "rule-b",
            "description": "delete alpha",
            "pattern": "alpha",
            "meta": {"source": "test"},
        },
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["data"] == {
        "patch_text": "",
        "modified_source": "",
        "applied_rules": [],
        "changes": [],
        "audit_trail": [],
    }
    assert result["errors"] == [
        {
            "type": "PatchConflictError",
            "error_type": "PatchConflictError",
            "message": "conflicting edits detected",
            "rule_id": "rule-a",
            "rule_index": 0,
            "details": {
                "rule_id": "rule-a",
                "rule_index": 0,
                "conflicting_rule_id": "rule-b",
                "conflicting_rule_index": 1,
                "rule_type": "replace",
                "conflicting_rule_type": "delete_regex",
                "span": [0, 6],
                "conflicting_span": [0, 6],
                "line_offsets": {
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 2,
                    "end_col": 0,
                },
                "conflicting_line_offsets": {
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 2,
                    "end_col": 0,
                },
            },
            "location": {
                "start_line": 1,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=[
            {"id": "rule-a", "type": "replace", "description": "replace alpha"},
            {"id": "rule-b", "type": "delete_regex", "description": "delete alpha"},
        ],
        applied_rules=[],
        anchor_resolutions=[
            {"id": "rule-a", "start": 0, "end": 6, "start_line": 1, "end_line": 2},
            {"id": "rule-b", "start": 0, "end": 6, "start_line": 1, "end_line": 2},
        ],
        changed_line_count=0,
        syntax_valid=None,
    )


def test_ambiguous_anchor_fails_with_patch_anchor_error():
    source = "alpha\nalpha\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-1",
            "description": "replace alpha",
            "anchor": "alpha",
            "replacement": "alpha-1",
            "meta": {"source": "test"},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["type"] == "PatchAnchorError"
    assert result["errors"][0]["rule_id"] == "rule-1"
    assert result["errors"][0]["details"]["anchor_search"]["match_count"] == 2


def test_syntax_breaking_insert_fails_with_patch_syntax_error():
    source = "def ok():\n    return 1\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-1",
            "description": "insert bad syntax",
            "anchor": "def ok():\n",
            "content": "def bad(\n",
            "meta": {"language": "python"},
        }
    ]

    result = _assert_deterministic(source, {"file_path": "example.py"}, rules)

    expected_source = "def ok():\ndef bad(\n    return 1\n"
    rule_summaries = [{"id": "rule-1", "type": "insert_after", "description": "insert bad syntax"}]
    applied_rules = [
        {
            "id": "rule-1",
            "type": "insert_after",
            "rationale": "insert bad syntax",
            "span": {"start": 10, "end": 10},
            "line_offsets": {
                "start_line": 2,
                "start_col": 0,
                "end_line": 2,
                "end_col": 0,
            },
        }
    ]
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
            "type": "PatchSyntaxError",
            "error_type": "PatchSyntaxError",
            "message": "Syntax check failed",
            "rule_id": None,
            "rule_index": None,
            "details": {
                "error_type": "SyntaxError",
                "message": "expected an indented block after function definition on line 1",
                "line": 2,
                "column": 1,
                "offset": 1,
                "rule_ids": ["rule-1"],
                "rule_count": 1,
            },
            "location": {
                "line": 2,
                "column": 1,
            },
        }
    ]
    assert result["meta"] == _expected_meta(
        source=source,
        rule_summaries=rule_summaries,
        applied_rules=applied_rules,
        anchor_resolutions=[{"id": "rule-1", "start": 10, "end": 10, "start_line": 2, "end_line": 2}],
        changed_line_count=1,
        syntax_valid=False,
    )


def test_rule_meta_can_force_syntax_validation():
    source = "def ok():\n    return 1\n"
    rules = [
        {
            "type": "insert_after",
            "id": "rule-1",
            "description": "insert bad syntax",
            "anchor": "def ok():\n",
            "content": "def bad(\n",
            "meta": {"validate_syntax": True},
        }
    ]

    result = _assert_deterministic(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["error_type"] == "PatchSyntaxError"
    details = result["errors"][0]["details"]
    assert details["line"] == 2
    assert details["column"] == 1
    assert details["message"] == "expected an indented block after function definition on line 1"


def test_meta_is_stable_across_repeated_runs():
    source = "alpha\nbravo\ncharlie\n"
    rules = [
        {
            "type": "replace",
            "id": "rule-replace",
            "description": "replace bravo",
            "anchor": "bravo\n",
            "replacement": "bravo-1\n",
            "meta": {"source": "test"},
        },
        {
            "type": "insert_after",
            "id": "rule-insert",
            "description": "insert beta",
            "anchor": "alpha\n",
            "content": "beta\n",
            "meta": {"source": "test"},
        },
    ]

    first = patch_generator.generate_patch(source, {}, rules)
    second = patch_generator.generate_patch(source, {}, rules)

    assert first["meta"] == second["meta"]
