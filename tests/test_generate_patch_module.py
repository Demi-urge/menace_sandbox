import pytest

import generate_patch
from menace.errors import PatchAnchorError, PatchRuleError
from menace_sandbox import patch_generator


def _replace_rule(*, rule_id: str, anchor: str = "alpha", replacement: str = "beta") -> dict:
    return {
        "type": "replace",
        "id": rule_id,
        "description": f"replace {anchor}",
        "anchor": anchor,
        "anchor_kind": "literal",
        "replacement": replacement,
        "meta": {"source": "unit"},
    }


def test_generate_patch_module_matches_patch_generator():
    source = "alpha\n"
    rules = [_replace_rule(rule_id="rule-1")]
    error_report = {"file_path": "example.txt"}

    module_result = generate_patch.generate_patch(source, error_report, rules)
    generator_result = patch_generator.generate_patch(source, error_report, rules)

    assert module_result == generator_result


def test_generate_patch_module_empty_rules_matches_generator():
    source = "alpha\n"
    rules = []

    module_result = generate_patch.generate_patch(source, {}, rules)
    generator_result = patch_generator.generate_patch(source, {}, rules)

    assert module_result == generator_result
    assert module_result["status"] == "error"
    assert module_result["errors"][0]["type"] == "PatchRuleError"


def test_generate_patch_module_raises_for_malformed_rules():
    with pytest.raises(PatchRuleError):
        generate_patch.generate_patch("alpha\n", {}, "bad-rules")  # type: ignore[arg-type]

    with pytest.raises(PatchRuleError):
        patch_generator.generate_patch("alpha\n", {}, "bad-rules")  # type: ignore[arg-type]


def test_generate_patch_module_raises_for_missing_anchor():
    source = "alpha\n"
    rules = [_replace_rule(rule_id="rule-1", anchor="missing")]

    with pytest.raises(PatchAnchorError):
        generate_patch.generate_patch(source, {}, rules)

    result = patch_generator.generate_patch(source, {}, rules)
    assert result["errors"][0]["type"] == "PatchAnchorError"
