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


def test_test_data_injection_and_expected_outputs(caplog):
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


def test_isolated_file_operations(tmp_path):
    runner = WorkflowSandboxRunner()

    outside_file = tmp_path / "should_not_exist.txt"

    def _workflow():
        with open(outside_file, "w") as fh:
            fh.write("data")

    runner.run(_workflow)

    assert not outside_file.exists()


def test_safe_mode_network_patching():
    runner = WorkflowSandboxRunner()

    def _workflow():
        import urllib.request

        urllib.request.urlopen("http://example.com")

    result = runner.run(_workflow, safe_mode=True)

    assert isinstance(result, RuntimeError)
    assert "network access disabled" in str(result)


def test_mock_injector_collects_telemetry():
    runner = WorkflowSandboxRunner()
    events: list[str] = []

    def _injector(_root):
        import builtins
        import pathlib

        original_open = builtins.open

        def _wrapped(file, mode="r", *a, **kw):
            events.append(pathlib.Path(file).name)
            return original_open(file, mode, *a, **kw)

        builtins.open = _wrapped  # type: ignore[assignment]
        return lambda: setattr(builtins, "open", original_open)

    runner.run(
        _sample_workflow,
        test_data={"input.txt": "hello"},
        mock_injectors=[_injector],
    )

    assert {"input.txt", "output.txt"} <= set(events)


def test_allowed_domain_access(monkeypatch):
    runner = WorkflowSandboxRunner()

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, *a, **kw: b"ok"
    )

    def _workflow():
        import urllib.request

        return urllib.request.urlopen("http://allowed.com")

    result = runner.run(
        _workflow, safe_mode=True, allowed_domains={"allowed.com"}
    )

    assert result == b"ok"


def test_allowed_file_writes(tmp_path):
    runner = WorkflowSandboxRunner()
    allowed = tmp_path / "allowed.txt"

    def _workflow():
        with open(allowed, "w") as fh:
            fh.write("data")

    runner.run(_workflow, allowed_files=[allowed])

    assert allowed.exists()
