import unittest
from unittest import mock

import mvp_codegen


class TestMvpCodegen(unittest.TestCase):
    def test_returns_string_for_empty_or_malformed_tasks(self) -> None:
        cases = [
            None,
            {},
            {"objective": None},
            {"objective": ""},
            {"objective": "   "},
            {"objective": 123},
            {"objective": "Make a tool."},
        ]

        for case in cases:
            with self.subTest(case=case):
                result = mvp_codegen.run_generation(case)  # type: ignore[arg-type]
                self.assertIsInstance(result, str)

    def test_model_wrapper_failure_returns_internal_error(self) -> None:
        class FailingWrapper:
            def __init__(self, model, tokenizer) -> None:
                raise RuntimeError("boom")

        task = {"objective": "Say hello.", "model": object(), "tokenizer": object()}
        with mock.patch.object(mvp_codegen.local_model_wrapper, "LocalModelWrapper", FailingWrapper):
            result = mvp_codegen.run_generation(task)

        self.assertIn("Internal error: code generation failed.", result)

    def test_blacklisted_imports_are_removed(self) -> None:
        class StubWrapper:
            def __init__(self, model, tokenizer) -> None:
                self.model = model
                self.tokenizer = tokenizer

            def generate(self, prompt, context_builder=None, max_new_tokens=256, do_sample=False):
                return "import os\nprint('ok')\nfrom sys import argv\nvalue = 1\n"

        task = {"objective": "Return a value.", "model": object(), "tokenizer": object()}
        with mock.patch.object(mvp_codegen.local_model_wrapper, "LocalModelWrapper", StubWrapper):
            result = mvp_codegen.run_generation(task)

        self.assertNotIn("import os", result)
        self.assertNotIn("from sys", result)
        self.assertIn("pass", result)
        self.assertIn("print('ok')", result)

    def test_oversized_outputs_are_truncated(self) -> None:
        class StubWrapper:
            def __init__(self, model, tokenizer) -> None:
                self.model = model
                self.tokenizer = tokenizer

            def generate(self, prompt, context_builder=None, max_new_tokens=256, do_sample=False):
                return "print('x')\n" * 600

        task = {"objective": "Print a line.", "model": object(), "tokenizer": object()}
        with mock.patch.object(mvp_codegen.local_model_wrapper, "LocalModelWrapper", StubWrapper):
            result = mvp_codegen.run_generation(task)

        self.assertLessEqual(len(result.encode("utf-8")), 4000)
        self.assertIn("print('x')", result)
        self.assertNotIn("Internal error", result)


if __name__ == "__main__":
    unittest.main()
