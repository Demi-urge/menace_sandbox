import unittest

import generate_patch
from menace.errors import PatchAnchorError, PatchConflictError, PatchRuleError


class GeneratePatchModuleTests(unittest.TestCase):
    def _replace_rule(self, *, rule_id, anchor="alpha", replacement="beta"):
        return {
            "type": "replace",
            "id": rule_id,
            "description": f"replace {anchor}",
            "anchor": anchor,
            "anchor_kind": "literal",
            "replacement": replacement,
            "meta": {"source": "unit"},
        }

    def _insert_rule(self, *, rule_id="rule-1", anchor="def ok():\n", content="def bad(\n"):
        return {
            "type": "insert_after",
            "id": rule_id,
            "description": "insert after anchor",
            "anchor": anchor,
            "anchor_kind": "literal",
            "content": content,
            "meta": {"source": "unit"},
        }

    def test_generate_patch_module_is_deterministic(self):
        source = "alpha\n"
        rules = [self._replace_rule(rule_id="rule-1")]

        first = generate_patch.generate_patch(source, {}, rules)
        second = generate_patch.generate_patch(source, {}, rules)

        self.assertEqual(first["data"]["patch_text"], second["data"]["patch_text"])
        self.assertEqual(first["meta"], second["meta"])

    def test_generate_patch_module_empty_rules_raises(self):
        source = "alpha\n"
        rules = []

        with self.assertRaises(PatchRuleError):
            generate_patch.generate_patch(source, {}, rules)

    def test_generate_patch_module_conflicting_edits_raise(self):
        source = "alpha\n"
        rules = [
            self._replace_rule(rule_id="rule-a", replacement="beta"),
            self._replace_rule(rule_id="rule-b", replacement="gamma"),
        ]

        with self.assertRaises(PatchConflictError):
            generate_patch.generate_patch(source, {}, rules)

    def test_generate_patch_module_invalid_anchor_raises(self):
        source = "alpha\n"
        rules = [self._replace_rule(rule_id="rule-1", anchor="missing")]

        with self.assertRaises(PatchAnchorError):
            generate_patch.generate_patch(source, {}, rules)

    def test_generate_patch_module_syntax_error_returns_structured_error(self):
        source = "def ok():\n    return 1\n"
        rules = [self._insert_rule()]

        module_result = generate_patch.generate_patch(
            source,
            {"file_path": "example.py"},
            rules,
            validate_syntax=True,
        )

        self.assertEqual(module_result["status"], "error")
        self.assertEqual(module_result["errors"][0]["type"], "PatchSyntaxError")
        self.assertIn("def bad(", module_result["data"]["modified_source"])


if __name__ == "__main__":
    unittest.main()
