import os
import tempfile

import pytest

import mvp_executor
import mvp_workflow
import mvp_workflow_legacy as legacy


def test_evaluate_result_deterministic():
    task = legacy.TaskSpec(
        objective="Validate deterministic ROI",
        constraints=["no network"],
    )
    exec_result = legacy.ExecutionResult(
        stdout="ok\n",
        stderr="",
        return_code=0,
        error="",
    )

    first = legacy.evaluate_result(task, exec_result)
    second = legacy.evaluate_result(task, exec_result)

    assert first == second
    assert first.roi_score == second.roi_score
    assert first.evaluation_error == second.evaluation_error


def test_execute_task_repeatable_fields():
    payload = {
        "objective": "Generate a short deterministic response",
        "constraints": ["no networking", "standard library only"],
    }

    results = [mvp_workflow.execute_task(payload) for _ in range(12)]
    baseline = results[0]

    for result in results[1:]:
        for field in ("roi_score", "success", "execution_error", "evaluation_error"):
            assert result[field] == baseline[field]


def test_execute_code_cleans_tempdir(monkeypatch: pytest.MonkeyPatch):
    created: list[object] = []
    original_tempdir = tempfile.TemporaryDirectory

    class TrackingTempDir:
        def __init__(self, *args, **kwargs):
            self._inner = original_tempdir(*args, **kwargs)
            self.name = self._inner.name
            self.cleaned = False
            created.append(self)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            result = self._inner.__exit__(exc_type, exc, tb)
            self.cleaned = not os.path.exists(self.name)
            return result

    monkeypatch.setattr(mvp_executor.tempfile, "TemporaryDirectory", TrackingTempDir)

    stdout, stderr = mvp_executor.execute_untrusted("print('ok')")

    assert created, "TemporaryDirectory was not invoked"
    tracked = created[0]
    assert tracked.cleaned
    assert not os.path.exists(tracked.name)
    assert "ok" in stdout
    assert stderr == ""
