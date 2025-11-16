from __future__ import annotations

import contextlib
import importlib
import json
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path


def _load_runner():
    # Stub meta logger to avoid heavy imports
    stub_meta = types.ModuleType("sandbox_runner.meta_logger")
    stub_meta._SandboxMetaLogger = None
    sys.modules["sandbox_runner.meta_logger"] = stub_meta

    # Ensure environment provides required hook
    env = sys.modules["sandbox_runner.environment"]
    env._patched_imports = contextlib.nullcontext

    return importlib.import_module("sandbox_runner.workflow_sandbox_runner").WorkflowSandboxRunner


def test_sandbox_execution_matches_fixture():
    """Sandbox metrics are stable for a simple workflow."""
    WorkflowSandboxRunner = _load_runner()
    runner = WorkflowSandboxRunner()

    def wf():
        return "ok"

    metrics = runner.run([wf], safe_mode=True)
    result = {
        "crash_count": metrics.crash_count,
        "modules": [{"name": m.name, "success": m.success} for m in metrics.modules],
    }

    expected = json.loads(
        resolve_path("tests/fixtures/regression/sandbox_metrics.json").read_text()
    )
    assert result == expected
