import os
import tempfile
import unittest
from unittest import mock

import mvp_workflow


def _find_generated_scripts(temp_dir: str) -> set[str]:
    matches: set[str] = set()
    for root, _dirs, files in os.walk(temp_dir):
        for name in files:
            if name == "generated_script.py":
                matches.add(os.path.join(root, name))
    return matches


class TestMvpWorkflow(unittest.TestCase):
    def test_execute_task_roi_deterministic(self) -> None:
        task = {
            "objective": "List the objective and constraints.",
            "constraints": ["No networking", "Standard library only"],
        }
        results = [mvp_workflow.execute_task(task) for _ in range(3)]
        roi_scores = [result["roi_score"] for result in results]
        self.assertTrue(all(score == roi_scores[0] for score in roi_scores))
        self.assertGreater(roi_scores[0], 0.0)
        self.assertTrue(all(result["success"] for result in results))

    def test_execute_task_timeout_zero_roi_no_traceback(self) -> None:
        fake_execution = {
            "execution_output": "STDOUT:\nTraceback (most recent call last):\n  boom\n",
            "errors": ["Traceback (most recent call last):\n  boom"],
            "stdout": "Traceback (most recent call last):\n  boom",
            "stderr": "Traceback (most recent call last):\n  boom",
            "exit_code": None,
            "timed_out": True,
        }
        with mock.patch.object(mvp_workflow, "_execute_code", return_value=fake_execution):
            result = mvp_workflow.execute_task({"objective": "Trigger timeout"})

        self.assertEqual(result["roi_score"], 0.0)
        self.assertFalse(result["success"])
        for field in ("execution_output", "execution_error", "evaluation_error"):
            self.assertNotIn("Traceback", result[field])

    def test_execute_task_cleans_temp_scripts(self) -> None:
        temp_dir = tempfile.gettempdir()
        before = _find_generated_scripts(temp_dir)

        result = mvp_workflow.execute_task({"objective": "Clean temp scripts"})

        after = _find_generated_scripts(temp_dir)
        self.assertTrue(result["success"])
        self.assertEqual(after - before, set())


if __name__ == "__main__":
    unittest.main()
