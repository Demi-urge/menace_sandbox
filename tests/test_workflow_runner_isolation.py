import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def WorkflowSandboxRunner():
    package_path = Path(__file__).resolve().parent.parent / "sandbox_runner"
    package = types.ModuleType("sandbox_runner")
    package.__path__ = [str(package_path)]
    sys.modules["sandbox_runner"] = package

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.workflow_runner", package_path / "workflow_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WorkflowSandboxRunner


@pytest.fixture()
def runner(WorkflowSandboxRunner):
    return WorkflowSandboxRunner()


def test_files_confined_to_temp_dir(tmp_path, runner):
    outside = tmp_path / "outside.txt"

    def workflow():
        with open(outside, "w") as fh:
            fh.write("content")

    runner.run(workflow)
    assert not outside.exists()


def test_safe_mode_blocks_network_and_files(monkeypatch, tmp_path, runner):
    outside = tmp_path / "leak.txt"
    called: list[str] = []

    def fake_urlopen(url, *a, **kw):
        called.append(url)
        return b"ok"

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    def workflow():
        with open(outside, "w") as fh:
            fh.write("data")
        import urllib.request

        urllib.request.urlopen("http://example.com")

    metrics = runner.run(workflow, safe_mode=True)

    assert not outside.exists()
    assert called == []  # our monkeypatch was bypassed by safe_mode patching
    mod = metrics.modules[0]
    assert mod.success is False
    assert mod.exception and "network access disabled" in mod.exception


def test_telemetry_includes_timing_and_memory(runner):
    def workflow():
        data = [i * i for i in range(10)]
        return sum(data)

    metrics = runner.run(workflow)
    mod = metrics.modules[0]

    assert mod.result == 285
    assert mod.duration >= 0
    assert isinstance(mod.memory_before, int)
    assert isinstance(mod.memory_after, int)
    assert mod.memory_after >= mod.memory_before
    assert mod.memory_delta == mod.memory_after - mod.memory_before
