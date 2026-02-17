from sandbox_runner.workflow_sandbox_runner import classify_failure_for_retry


def test_classify_failure_for_retry_expected_misuse():
    classification, policy = classify_failure_for_retry(
        'SANDBOX_MISUSE_EVENT={"kind":"expected_user_misuse"}'
    )
    assert classification == "expected_simulation_misuse"
    assert policy == "handled_expected_failure"


def test_classify_failure_for_retry_policy_enforcement():
    classification, policy = classify_failure_for_retry(PermissionError("forbidden path access"))
    assert classification == "policy_enforcement_event"
    assert policy == "policy_enforcement"


def test_classify_failure_for_retry_runtime_regression():
    classification, policy = classify_failure_for_retry(RuntimeError("unexpected crash"))
    assert classification == "runtime_regression"
    assert policy == "retry_runtime_regression"
