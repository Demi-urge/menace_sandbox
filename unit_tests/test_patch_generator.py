import unittest
from unittest import mock

from menace.errors import PatchAnchorError, PatchConflictError, PatchRuleError, PatchSyntaxError
from menace_sandbox import patch_generator


class PatchGeneratorTests(unittest.TestCase):
    def _replace_rule(self, *, rule_id="rule-1", anchor="alpha", replacement="beta"):
        return patch_generator.ReplaceRule(
            rule_id=rule_id,
            description="replace alpha",
            anchor=anchor,
            replacement=replacement,
            anchor_kind="literal",
            meta={"source": "unit"},
        )

    def _insert_rule(self, *, rule_id="rule-1", anchor="def ok():\n", content="def bad(\n"):
        return patch_generator.InsertAfterRule(
            rule_id=rule_id,
            description="insert after anchor",
            anchor=anchor,
            content=content,
            anchor_kind="literal",
            meta={"language": "python"},
        )

    def test_identical_inputs_produce_identical_patch_text_and_meta(self):
        source = "alpha\n"
        rules = [self._replace_rule()]

        with mock.patch.object(patch_generator.time, "monotonic", return_value=1.0):
            first = patch_generator.generate_patch(source, {}, rules)
            second = patch_generator.generate_patch(source, {}, rules)

        self.assertEqual(first["data"]["patch_text"], second["data"]["patch_text"])
        self.assertEqual(first["meta"], second["meta"])

    def test_empty_rules_return_structured_error_payload(self):
        with self.assertRaises(PatchRuleError):
            patch_generator.generate_patch("alpha\n", {}, [])

    def test_invalid_rules_raise_menace_errors(self):
        with self.assertRaises(PatchRuleError):
            patch_generator.validate_rules([object()])

    def test_contradictory_rules_raise_menace_errors(self):
        source = "alpha\n"
        rules = [
            self._replace_rule(rule_id="rule-a", replacement="alpha-1"),
            self._replace_rule(rule_id="rule-b", replacement="alpha-2"),
        ]

        with self.assertRaises(PatchConflictError):
            patch_generator.apply_rules(source, rules)

    def test_missing_anchor_raises_patch_anchor_error(self):
        source = "alpha\n"
        rules = [self._replace_rule(anchor="missing")]

        with self.assertRaises(PatchAnchorError):
            patch_generator.apply_rules(source, rules)

    def test_syntax_breaking_edits_rejected_with_validation_enabled(self):
        source = "def ok():\n    return 1\n"
        rules = [self._insert_rule()]

        result = patch_generator.generate_patch(
            source,
            {"file_path": "example.py"},
            rules,
            validate_syntax=True,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["data"], {})
        self.assertEqual(result["errors"][0]["error_type"], "PatchSyntaxError")


if __name__ == "__main__":
    unittest.main()
