import logging
import os

import pytest

from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def test_run_logs_stat_error(caplog):
    runner = WorkflowSandboxRunner()

    def wf():
        os.stat("/definitely/missing/path")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            runner.run(wf)
    assert any("/definitely/missing/path" in r.message for r in caplog.records)
