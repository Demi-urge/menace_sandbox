from menace_sandbox import patch_generator


def test_generate_patch_deterministic_output():
    source = "alpha\nbravo\n"
    rules = [
        {"type": "replace", "line": 1, "content": "alpha-1", "id": "rule-1"},
        {"type": "insert_after", "line": 2, "content": "charlie", "id": "rule-2"},
    ]
    error_report = {"file_path": "module.py"}

    first = patch_generator.generate_patch(source, error_report, rules)
    second = patch_generator.generate_patch(source, error_report, rules)

    assert first == second
    assert list(first.keys()) == ["status", "data", "errors", "meta"]
    assert list(first["meta"].keys()) == ["applied_rules", "line_counts", "change_summary"]


def test_generate_patch_validation_errors_returned():
    source = "alpha\nbravo\n"
    error_report = {}
    cases = [
        ([{"type": "unknown", "line": 1, "content": "x"}], "Unknown rule type"),
        (["not-a-mapping"], "rule must be a mapping"),
        ([{"type": "replace", "line": "one", "content": "x"}], "rule line must be an integer"),
    ]

    for rules, message in cases:
        result = patch_generator.generate_patch(source, error_report, rules)
        assert result["status"] == "failed"
        assert result["errors"]
        error = result["errors"][0]
        assert error["error_type"] == "ValidationError"
        assert error["message"] == message


def test_generate_patch_impossible_anchor_fails():
    source = "alpha\nbravo\n"
    rules = [{"type": "insert_after", "line": 99, "content": "charlie"}]

    result = patch_generator.generate_patch(source, {}, rules)

    assert result["status"] == "failed"
    assert result["errors"][0]["error_type"] == "ValidationError"
    assert result["errors"][0]["message"] == "insert anchor is out of range"


def test_generate_patch_conflicting_replacements_fail():
    source = "alpha\nbravo\n"
    rules = [
        {"type": "replace", "line": 1, "content": "alpha-1", "id": "rule-a"},
        {"type": "replace", "line": 1, "content": "alpha-2", "id": "rule-b"},
    ]

    result = patch_generator.generate_patch(source, {}, rules)

    assert result["status"] == "failed"
    error = result["errors"][0]
    assert error["error_type"] == "ValidationError"
    assert error["message"] == "replace conflicts with previous rule"
    assert error["details"]["conflicting_rule_id"] == "rule-a"


def test_generate_patch_empty_rules_noop():
    source = "alpha\nbravo\n"

    result = patch_generator.generate_patch(source, {}, [])

    assert result["status"] == "success"
    assert result["data"]["diff"] == ""
    assert result["data"]["source"] == source
    assert result["errors"] == []
    assert result["meta"] == {
        "applied_rules": [],
        "line_counts": {"before": 2, "after": 2},
        "change_summary": {"inserted": 0, "deleted": 0, "replaced": 0},
    }


def test_generate_patch_syntax_error_reported():
    source = "def ok():\n    return 1\n"
    rules = [{"type": "insert_before", "line": 1, "content": "def bad("}]

    result = patch_generator.generate_patch(source, {"file_path": "example.py"}, rules)

    assert result["status"] == "failed"
    assert result["errors"]
    error = result["errors"][0]
    assert error["error_type"] == "ValidationError"
    assert error["message"] == "Syntax check failed"
    assert error["details"]["error_type"] == "SyntaxError"
