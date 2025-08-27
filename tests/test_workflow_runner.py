from pathlib import Path
import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

package_path = Path(__file__).resolve().parent.parent / "sandbox_runner"
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_runner", package_path / "workflow_runner.py"
)
workflow_runner = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(workflow_runner)
WorkflowSandboxRunner = workflow_runner.WorkflowSandboxRunner


def test_writes_confined_to_temp_dir(tmp_path):
    outside = tmp_path / "should_not_persist.txt"

    def writer():
        with open(outside, "w") as fh:
            fh.write("data")

    runner = WorkflowSandboxRunner()
    runner.run(writer)

    assert not outside.exists()


def test_safe_mode_blocks_network_and_records_error():
    target = Path("unsafe_output.txt")

    def network_and_file():
        with open(target, "w") as fh:
            fh.write("data")
        import urllib.request

        urllib.request.urlopen("http://example.com")

    runner = WorkflowSandboxRunner()
    result, telemetry = runner.run(network_and_file, safe_mode=True)

    assert result is None
    assert telemetry["success"] is False
    assert "network access disabled" in telemetry["error"]
    assert not target.exists()


def test_prepopulated_data_and_telemetry():
    def reader():
        return Path("in.txt").read_text()

    runner = WorkflowSandboxRunner()
    result, telemetry = runner.run(reader, test_data={"in.txt": "hello"})

    assert result == "hello"
    assert telemetry["success"] is True
    assert "duration" in telemetry and "memory_delta" in telemetry
    assert runner.telemetry == telemetry

