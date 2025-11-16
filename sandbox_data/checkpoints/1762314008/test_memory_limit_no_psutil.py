import importlib.util
import sys

import pytest

from dynamic_path_router import resolve_path, repo_root

# Ensure repository root on path for sandbox_settings import
sys.path.append(str(repo_root()))

module_path = resolve_path("workflow_sandbox_runner.py")  # path-ignore
spec = importlib.util.spec_from_file_location("workflow_sandbox_runner", module_path)
workflow_sandbox_runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workflow_sandbox_runner
spec.loader.exec_module(workflow_sandbox_runner)  # type: ignore[arg-type]
WorkflowSandboxRunner = workflow_sandbox_runner.WorkflowSandboxRunner


def dummy():
    return 42


def test_memory_limit_requires_psutil(monkeypatch):
    monkeypatch.setattr(workflow_sandbox_runner, "psutil", None)
    runner = WorkflowSandboxRunner()
    with pytest.raises(RuntimeError):
        runner.run(dummy, memory_limit=1024)
