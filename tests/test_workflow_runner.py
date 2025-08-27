import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

called = {}

# Provide a fake environment module so importing workflow_runner doesn't pull heavy deps
fake_env = types.ModuleType("sandbox_runner.environment")

def fake_generate_input_stubs(count=None, *, target=None, strategy=None, providers=None):
    called['target'] = target
    called['providers'] = providers
    return [{'value': 7}]

fake_env.generate_input_stubs = fake_generate_input_stubs
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(Path(__file__).resolve().parent.parent / "sandbox_runner")]
sys.modules["sandbox_runner"] = package
sys.modules["sandbox_runner.environment"] = fake_env

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_runner",
    Path(__file__).resolve().parent.parent / "sandbox_runner" / "workflow_runner.py",
)
workflow_runner = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(workflow_runner)


def test_runner_generates_stubs_and_logs(caplog):
    def sample_module(value):
        return value + 1

    provider = lambda stubs, ctx: stubs
    runner = workflow_runner.WorkflowSandboxRunner([sample_module], stub_providers=[provider])

    with caplog.at_level(logging.INFO):
        metrics = runner.run()

    assert called['target'] is sample_module
    assert called['providers'] == [provider]
    assert metrics['modules'][0]['stub'] == {'value': 7}
    assert metrics['modules'][0]['result'] == 8
    assert any('stub' in r.message and 'value' in r.message for r in caplog.records)


def test_runner_records_telemetry_and_crashes():
    def good():
        return 1

    def bad():
        raise RuntimeError('boom')

    runner = workflow_runner.WorkflowSandboxRunner([good, bad], safe_mode=True)
    metrics = runner.run()

    assert len(metrics['modules']) == 2
    first = metrics['modules'][0]
    assert 'duration' in first and 'memory_delta' in first and 'memory_peak' in first
    assert metrics['modules'][1]['exception'] is True
    assert runner.crash_counts['bad'] == 1
    assert metrics['crash_counts']['bad'] == 1
