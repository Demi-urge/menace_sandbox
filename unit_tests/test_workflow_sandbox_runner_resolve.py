import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
package_path = ROOT / "sandbox_runner"
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner", package_path / "workflow_sandbox_runner.py"
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner


def test_resolve_inside(tmp_path):
    runner = WorkflowSandboxRunner()
    resolved = runner._resolve(tmp_path, "foo.txt")
    assert resolved == (tmp_path / "foo.txt").resolve()


def test_resolve_absolute_path(tmp_path):
    runner = WorkflowSandboxRunner()
    inside = (tmp_path / "bar.txt").resolve()
    resolved = runner._resolve(tmp_path, inside)
    assert resolved == inside


def test_resolve_prevents_escape(tmp_path):
    runner = WorkflowSandboxRunner()
    with pytest.raises(RuntimeError):
        runner._resolve(tmp_path, "../escape.txt")
