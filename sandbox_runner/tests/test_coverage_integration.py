import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def _sample_module():
    def inner():
        return 42
    return inner()


def test_module_coverage_reporting():
    runner = WorkflowSandboxRunner()
    metrics = runner.run(_sample_module, use_subprocess=False)
    mod = metrics.modules[0]
    assert isinstance(mod.coverage_files, list)
    assert isinstance(mod.coverage_functions, list)

