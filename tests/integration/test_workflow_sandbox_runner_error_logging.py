import logging
import os
from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def test_run_logs_stat_error(caplog):
    runner = WorkflowSandboxRunner()

    def wf():
        os.stat("/definitely/missing/path")

    with caplog.at_level(logging.ERROR):
        try:
            runner.run(wf)
        except Exception:
            pass
    assert any("unexpected error" in r.message for r in caplog.records)
