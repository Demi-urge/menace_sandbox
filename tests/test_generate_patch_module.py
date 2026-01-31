import pytest

import generate_patch
from menace.errors import PatchRuleError


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


def test_generate_patch_module_determinism():
    source = "alpha\n"
    rules = [_replace_rule(rule_id="rule-1")]

    first = generate_patch.generate_patch(source, {}, rules)
    second = generate_patch.generate_patch(source, {}, rules)

    assert first["data"]["patch_text"] == second["data"]["patch_text"]
    assert first["meta"] == second["meta"]


def test_generate_patch_module_empty_rules_returns_structured_error():
    source = "alpha\n"
    rules = []

    result = generate_patch.generate_patch(source, {}, rules)

    assert result["status"] == "error"
    assert result["errors"][0]["type"] == "PatchRuleError"


def test_generate_patch_module_raises_for_malformed_rules():
    with pytest.raises(PatchRuleError):
        generate_patch.generate_patch("alpha\n", {}, "bad-rules")  # type: ignore[arg-type]


def test_generate_patch_module_raises_for_missing_anchor():
    source = "alpha\n"
    rules = [_replace_rule(rule_id="rule-1", anchor="missing")]

    result = generate_patch.generate_patch(source, {}, rules)
    assert result["errors"][0]["type"] == "PatchAnchorError"
