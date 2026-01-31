import unittest

import generate_patch
from menace_sandbox import patch_generator


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

    def test_generate_patch_module_matches_patch_generator(self):
        source = "alpha\n"
        rules = [
            {
                "type": "replace",
                "id": "rule-1",
                "description": "replace alpha",
                "anchor": "alpha",
                "anchor_kind": "literal",
                "replacement": "beta",
                "meta": {"source": "unit"},
            }
        ]
        error_report = {"file_path": "example.txt"}

        module_result = generate_patch.generate_patch(source, error_report, rules)
        generator_result = patch_generator.generate_patch(source, error_report, rules)

        self.assertEqual(module_result, generator_result)

    def test_generate_patch_module_empty_rules_returns_structured_error(self):
        source = "alpha\n"
        rules = []

        module_result = generate_patch.generate_patch(source, {}, rules)
        generator_result = patch_generator.generate_patch(source, {}, rules)

        self.assertEqual(module_result, generator_result)
        self.assertEqual(module_result["status"], "error")
        self.assertEqual(module_result["errors"][0]["type"], "PatchRuleError")

    def test_generate_patch_module_conflicting_edits_return_structured_error(self):
        source = "alpha\n"
        rules = [
            self._replace_rule(rule_id="rule-a", replacement="beta"),
            self._replace_rule(rule_id="rule-b", replacement="gamma"),
        ]

        module_result = generate_patch.generate_patch(source, {}, rules)
        generator_result = patch_generator.generate_patch(source, {}, rules)

        self.assertEqual(module_result, generator_result)
        self.assertEqual(module_result["status"], "error")
        self.assertEqual(module_result["errors"][0]["type"], "PatchConflictError")

    def test_generate_patch_module_invalid_anchor_returns_structured_error(self):
        source = "alpha\n"
        rules = [self._replace_rule(rule_id="rule-1", anchor="missing")]

        module_result = generate_patch.generate_patch(source, {}, rules)
        generator_result = patch_generator.generate_patch(source, {}, rules)

        self.assertEqual(module_result, generator_result)
        self.assertEqual(module_result["status"], "error")
        self.assertEqual(module_result["errors"][0]["type"], "PatchAnchorError")

    def test_generate_patch_module_syntax_error_returns_structured_error(self):
        source = "def ok():\n    return 1\n"
        rules = [self._insert_rule()]

        module_result = generate_patch.generate_patch(
            source,
            {"file_path": "example.py"},
            rules,
            validate_syntax=True,
        )
        generator_result = patch_generator.generate_patch(
            source,
            {"file_path": "example.py"},
            rules,
            validate_syntax=True,
        )

        self.assertEqual(module_result, generator_result)
        self.assertEqual(module_result["status"], "error")
        self.assertEqual(module_result["errors"][0]["type"], "PatchSyntaxError")


if __name__ == "__main__":
    unittest.main()
