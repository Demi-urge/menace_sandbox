import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

called: dict[str, object] = {}

# Provide a lightweight environment module so importing workflow_runner avoids heavy deps
fake_env = types.ModuleType("sandbox_runner.environment")


def fake_generate_input_stubs(
    count=None, *, target=None, strategy=None, providers=None
):
    called["target"] = target
    called["providers"] = providers
    return [{"value": 3}]


fake_env.generate_input_stubs = fake_generate_input_stubs
package = types.ModuleType("sandbox_runner")
package.__path__ = [
    str(Path(__file__).resolve().parent.parent / "sandbox_runner")
]
sys.modules["sandbox_runner"] = package
sys.modules["sandbox_runner.environment"] = fake_env

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_runner",
    Path(__file__).resolve().parent.parent
    / "sandbox_runner"
    / "workflow_runner.py",
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

    runner = WorkflowSandboxRunner([writer])
    runner.run()

    assert not outside.exists()


def test_safe_mode_blocks_network_and_file_saves():
    target = Path("unsafe_output.txt")

    def network_and_file():
        with open(target, "w") as fh:
            fh.write("data")
        import urllib.request

        urllib.request.urlopen("http://example.com")

    runner = WorkflowSandboxRunner([network_and_file], safe_mode=True)
    metrics = runner.run()

    assert metrics["modules"][0]["exception"] is True
    assert "network access disabled" in metrics["modules"][0]["error"]
    assert not target.exists()


def test_stub_inputs_and_telemetry_and_crash_counts():
    called.clear()

    def good(value):
        return value + 1

    def bad():
        raise RuntimeError("boom")

    def provider(stubs, ctx):
        return stubs

    runner = WorkflowSandboxRunner([good, bad], safe_mode=True, stub_providers=[provider])
    metrics = runner.run()

    assert called["target"] is good
    assert called["providers"] == [provider]

    first = metrics["modules"][0]
    assert first["stub"] == {"value": 3}
    assert first["result"] == 4
    assert "duration" in first and "memory_delta" in first and "memory_peak" in first

    assert metrics["modules"][1]["exception"] is True
    assert runner.crash_counts["bad"] == 1
    assert metrics["crash_counts"]["bad"] == 1
