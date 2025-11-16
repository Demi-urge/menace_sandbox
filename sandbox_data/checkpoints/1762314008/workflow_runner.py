from __future__ import annotations

import warnings

from .workflow_sandbox_runner import WorkflowSandboxRunner

warnings.warn(
    "sandbox_runner.workflow_runner is deprecated; use "
    "sandbox_runner.workflow_sandbox_runner.WorkflowSandboxRunner instead",
    DeprecationWarning,
    stacklevel=2,
)

WorkflowRunner = WorkflowSandboxRunner

__all__ = ["WorkflowRunner", "WorkflowSandboxRunner"]
