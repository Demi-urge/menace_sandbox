import unittest

import generate_patch
from menace_sandbox import patch_generator


class GeneratePatchModuleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
