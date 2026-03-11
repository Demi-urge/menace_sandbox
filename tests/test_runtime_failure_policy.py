from runtime_failure_policy import RuntimeFailureReason, classify_runtime_failure


def test_classify_runtime_failure_critical_buckets():
    coding = classify_runtime_failure(reason_code=RuntimeFailureReason.SELF_CODING_ENGINE_CRASH)
    assert coding.category == "critical"
    assert coding.reason == RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value
    assert coding.should_exit is True

    cycle = classify_runtime_failure(reason_code=RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT)
    assert cycle.category == "critical"
    assert cycle.reason == RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value
    assert cycle.should_exit is True


def test_classify_runtime_failure_self_improvement_and_learning_variants_are_critical():
    learning = classify_runtime_failure(reason_code=RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT)
    assert learning.category == "critical"
    assert learning.reason == RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value
    assert learning.should_exit is True

    improvement = classify_runtime_failure(reason_code=RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT)
    assert improvement.category == "critical"
    assert improvement.reason == RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value
    assert improvement.should_exit is True

    managed_worker = classify_runtime_failure(reason_code=RuntimeFailureReason.SELF_CODING_ENGINE_CRASH)
    assert managed_worker.category == "critical"
    assert managed_worker.reason == RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value
    assert managed_worker.should_exit is True


def test_classify_runtime_failure_non_critical_buckets():
    telemetry = classify_runtime_failure(reason_code=RuntimeFailureReason.TELEMETRY_FAILURE)
    assert telemetry.category == "non_critical"
    assert telemetry.reason == RuntimeFailureReason.TELEMETRY_FAILURE.value
    assert telemetry.should_exit is False

    optional_dep = classify_runtime_failure(reason_code=RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE)
    assert optional_dep.category == "non_critical"
    assert optional_dep.reason == RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE.value
    assert optional_dep.should_exit is False

    api = classify_runtime_failure(reason_code=RuntimeFailureReason.INTEGRATION_OR_API_ERROR)
    assert api.category == "non_critical"
    assert api.reason == RuntimeFailureReason.INTEGRATION_OR_API_ERROR.value
    assert api.should_exit is False



def test_classify_runtime_failure_legacy_substring_fallback(caplog):
    caplog.set_level("WARNING")
    fallback = classify_runtime_failure(event="self-improvement cycle thread crashed")
    assert fallback.reason == RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value
    assert "legacy runtime failure substring classification used" in caplog.text
