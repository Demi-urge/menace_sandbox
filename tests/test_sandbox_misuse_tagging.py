from pathlib import Path


def test_user_misuse_injection_uses_structured_forbidden_path_tag():
    src = Path("sandbox_runner/environment.py").read_text(encoding="utf-8")
    assert "_MISUSE_FORBIDDEN_PATH_TAG='SIMULATED_MISUSE_FORBIDDEN_PATH'" in src
    assert "_emit_misuse_event(_MISUSE_FORBIDDEN_PATH_TAG + ': /root/forbidden')" in src


def test_repo_section_aggregation_treats_tagged_misuse_as_expected_artifact():
    src = Path("sandbox_runner/environment.py").read_text(encoding="utf-8")
    assert "tagged_expected_misuse" in src
    assert "res.get(\"expected_scenario_fault\") == \"user_misuse\"" in src
    assert '"SIMULATED_MISUSE_FORBIDDEN_PATH" in str(res.get("stderr", ""))' in src
    assert "and not tagged_expected_misuse" in src


def test_real_guard_violation_signature_is_unchanged():
    from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner

    def step():
        with open('/root/forbidden', 'r'):
            pass

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True, use_subprocess=False, subprocess_guard=True)

    assert metrics.crash_count == 1
    exc = metrics.modules[0].exception or ""
    assert "error: forbidden path access" in exc
