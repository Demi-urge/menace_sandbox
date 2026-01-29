import os
import tempfile
import unittest
from unittest import mock

import mvp_workflow


class TestMvpWorkflow(unittest.TestCase):
    def _strip_timestamps(self, payload: dict[str, object]) -> dict[str, object]:
        cleaned = dict(payload)
        cleaned.pop("timestamps", None)
        return cleaned

    def test_execute_task_is_deterministic(self) -> None:
        task = {"objective": "Echo payload for testing", "constraints": ["fast"]}

        first = mvp_workflow.execute_task(task)
        second = mvp_workflow.execute_task(task)

        self.assertEqual(first["roi_score"], second["roi_score"])
        self.assertEqual(first["success"], second["success"])
        self.assertEqual(
            self._strip_timestamps(first),
            self._strip_timestamps(second),
        )

    def test_execute_task_missing_objective(self) -> None:
        result = mvp_workflow.execute_task({})

        self.assertFalse(result["success"])
        self.assertTrue(result["errors"])

    def test_execute_task_timeout(self) -> None:
        slow_code = """\
import time

def main() -> None:
    time.sleep(10)

if __name__ == '__main__':
    main()
"""
        captured: dict[str, object] = {}
        original_execute = mvp_workflow._execute_code

        def _capture_execute(code: str, timeout_s: float) -> dict[str, object]:
            result = original_execute(code, timeout_s)
            captured.update(result)
            return result

        with mock.patch.object(
            mvp_workflow,
            "_generate_code",
            return_value={"generated_code": slow_code, "errors": []},
        ):
            with mock.patch.object(mvp_workflow, "_execute_code", side_effect=_capture_execute):
                result = mvp_workflow.execute_task({"objective": "Sleep forever"})

        self.assertFalse(result["success"])
        self.assertIn("execution timed out", result["errors"])
        self.assertTrue(captured.get("timed_out"))

    def test_execute_task_cleans_temp_files(self) -> None:
        temp_dir = tempfile.gettempdir()

        def _find_generated_scripts() -> set[str]:
            matches: set[str] = set()
            for root, _, files in os.walk(temp_dir):
                if "generated_script.py" in files:
                    matches.add(os.path.join(root, "generated_script.py"))
            return matches

        before = _find_generated_scripts()
        mvp_workflow.execute_task({"objective": "Check temp cleanup"})
        after = _find_generated_scripts()

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
