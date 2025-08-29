from pathlib import Path

from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def test_unmocked_network_raises_runtimeerror():
    def step():
        import urllib.request

        urllib.request.urlopen("http://example.com")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "network access disabled" in (metrics.modules[0].exception or "")


def test_absolute_path_write_raises_runtimeerror(tmp_path):
    outside = Path("/tmp/outside_escape.txt")

    def step():
        outside.write_text("data")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "file write disabled" in (metrics.modules[0].exception or "")
    assert not outside.exists()
