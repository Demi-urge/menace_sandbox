import importlib.util
import logging
from pathlib import Path

module_path = (
    Path(__file__).resolve().parent.parent
    / "sandbox_runner"
    / "workflow_sandbox_runner.py"
)
spec = importlib.util.spec_from_file_location(
    "workflow_sandbox_runner", module_path
)
workflow_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # for type checkers
spec.loader.exec_module(workflow_module)  # type: ignore[attr-defined]
WorkflowSandboxRunner = workflow_module.WorkflowSandboxRunner


def _sample_workflow():
    with open("input.txt") as fh:
        data = fh.read()
    with open("output.txt", "w") as fh:
        fh.write(data.upper())


def test_prepopulated_inputs_and_expected_outputs(caplog):
    runner = WorkflowSandboxRunner()
    with caplog.at_level(logging.WARNING):
        runner.run(
            _sample_workflow,
            test_data={"input.txt": "hello"},
            expected_outputs={"output.txt": "HELLO"},
        )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_expected_output_mismatch_logged(caplog):
    runner = WorkflowSandboxRunner()
    with caplog.at_level(logging.WARNING):
        runner.run(
            _sample_workflow,
            test_data={"input.txt": "hello"},
            expected_outputs={"output.txt": "WRONG"},
        )
    assert any("output mismatch" in r.message for r in caplog.records)
