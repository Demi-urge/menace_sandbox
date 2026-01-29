import datetime
import json
import os
import tempfile
import unittest
from unittest import mock

import mvp_workflow


class FixedDateTime(datetime.datetime):
    """Datetime stub with controllable now() values."""

    _now_values = []

    @classmethod
    def now(cls, tz=None):
        if not cls._now_values:
            raise AssertionError("No more fixed datetime values available")
        value = cls._now_values.pop(0)
        if tz is not None:
            return value.astimezone(tz)
        return value


class TestMvpWorkflow(unittest.TestCase):
    def test_deterministic_roi_for_same_input(self):
        base_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        end_time = base_time + datetime.timedelta(milliseconds=120)
        FixedDateTime._now_values = [base_time, end_time, base_time, end_time]

        with mock.patch.object(mvp_workflow.datetime, "datetime", FixedDateTime):
            result_one = mvp_workflow.execute_task({"objective": "ping"})
            result_two = mvp_workflow.execute_task({"objective": "ping"})

        self.assertEqual(result_one["roi_score"], result_two["roi_score"])
        self.assertEqual(result_one["generated_code"], result_two["generated_code"])

    def test_missing_objective_returns_soft_error(self):
        result = mvp_workflow.execute_task({})

        self.assertFalse(result["success"])
        self.assertTrue(any("objective must be a non-empty string" in err for err in result["errors"]))

    def test_empty_generated_code_captures_error(self):
        with mock.patch.object(mvp_workflow, "_generate_code", return_value=""):
            result = mvp_workflow.execute_task({"objective": "ignored"})

        self.assertFalse(result["success"])
        self.assertTrue(any("code is empty" in err for err in result["errors"]))

    def test_invalid_generated_code_captures_error(self):
        with mock.patch.object(mvp_workflow, "_generate_code", return_value="def bad("):
            result = mvp_workflow.execute_task({"objective": "ignored"})

        self.assertFalse(result["success"])
        self.assertTrue(any("syntax error" in err for err in result["errors"]))

    def test_timeout_and_tempfile_cleanup(self):
        created_paths = []
        original_named_tempfile = tempfile.NamedTemporaryFile

        def tracking_named_tempfile(*args, **kwargs):
            tmp = original_named_tempfile(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        loop_code = "while True: pass"
        with mock.patch.object(mvp_workflow.tempfile, "NamedTemporaryFile", tracking_named_tempfile):
            stdout, errors = mvp_workflow._execute_code(loop_code, timeout_s=0.05)

        self.assertEqual("", stdout)
        self.assertIn("timeout", errors)
        for path in created_paths:
            self.assertFalse(os.path.exists(path), f"temp file not removed: {path}")

    def test_output_is_json_serializable_with_required_types(self):
        result = mvp_workflow.execute_task({"objective": "serialize"})
        payload = json.dumps(result)

        self.assertIsInstance(payload, str)
        self.assertIsInstance(result["generated_code"], str)
        self.assertIsInstance(result["execution_output"], str)
        self.assertIsInstance(result["errors"], list)
        self.assertIsInstance(result["roi_score"], float)
        self.assertIsInstance(result["timestamps"], dict)
        self.assertIsInstance(result["timestamps"]["started_at"], str)
        self.assertIsInstance(result["timestamps"]["finished_at"], str)
        self.assertIsInstance(result["timestamps"]["duration_ms"], int)
        self.assertIsInstance(result["success"], bool)


if __name__ == "__main__":
    unittest.main()
