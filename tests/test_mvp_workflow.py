import os
import tempfile
import unittest
from unittest import mock

import mvp_workflow


class TestMvpWorkflow(unittest.TestCase):
    def test_validation_missing_objective(self):
        result = mvp_workflow.execute_task({})

        self.assertFalse(result["success"])
        self.assertEqual(result["errors"], ["objective must be a non-empty string"])
        self.assertIsNone(result["generated_code"])
        self.assertIsNone(result["execution_output"])

    def test_validation_empty_objective(self):
        result = mvp_workflow.execute_task({"objective": "  "})

        self.assertFalse(result["success"])
        self.assertEqual(result["errors"], ["objective must be a non-empty string"])
        self.assertIsNone(result["generated_code"])
        self.assertIsNone(result["execution_output"])

    def test_deterministic_execution(self):
        task = {"objective": "Run a deterministic workflow", "constraints": {"mode": "test"}}

        first = mvp_workflow.execute_task(task)
        second = mvp_workflow.execute_task(task)

        for key in ("generated_code", "execution_output", "errors", "roi_score", "success"):
            self.assertEqual(first[key], second[key])

    def test_forbidden_import_blocks_execution(self):
        code = "import os\nprint('blocked')\n"
        original_checker = mvp_workflow._is_stdlib_module

        def fake_checker(module_name: str) -> bool:
            if module_name == "os":
                return False
            return original_checker(module_name)

        with mock.patch("mvp_workflow._is_stdlib_module", side_effect=fake_checker):
            result = mvp_workflow._execute_code(code, timeout_s=1.0)

        self.assertEqual(result["error"], "forbidden import: os")
        self.assertEqual(result["stdout"], "")

    def test_timeout_reports_failure_and_cleans_tempfile(self):
        original_execute = mvp_workflow._execute_code
        created_paths = []
        original_tempfile = tempfile.NamedTemporaryFile

        def tracking_named_tempfile(*args, **kwargs):
            tmp = original_tempfile(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        def fast_execute(code: str, timeout_s: float) -> dict:
            return original_execute(code, timeout_s=0.1)

        with mock.patch("mvp_workflow._generate_code", return_value={"code": "while True: pass", "error": None}):
            with mock.patch("mvp_workflow._execute_code", side_effect=fast_execute):
                with mock.patch("mvp_workflow.tempfile.NamedTemporaryFile", side_effect=tracking_named_tempfile):
                    result = mvp_workflow.execute_task({"objective": "loop"})

        self.assertFalse(result["success"])
        self.assertIn("timeout", result["errors"])
        self.assertTrue(created_paths)
        for path in created_paths:
            self.assertFalse(
                os.path.exists(path),
                msg=f"Temporary file should be removed: {path}",
            )

    def test_cleanup_removes_tempfile_after_success(self):
        created_paths = []
        original_tempfile = tempfile.NamedTemporaryFile

        def tracking_named_tempfile(*args, **kwargs):
            tmp = original_tempfile(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        with mock.patch("mvp_workflow.tempfile.NamedTemporaryFile", side_effect=tracking_named_tempfile):
            result = mvp_workflow._execute_code("print('ok')", timeout_s=1.0)

        self.assertIsNone(result["error"])
        self.assertTrue(created_paths)
        for path in created_paths:
            self.assertFalse(
                os.path.exists(path),
                msg=f"Temporary file should be removed: {path}",
            )


if __name__ == "__main__":
    unittest.main()
