from runtime_failure_policy import classify_runtime_failure


def test_classify_runtime_failure_critical_buckets():
    coding = classify_runtime_failure(component="self_coding_engine")
    assert coding.category == "critical"
    assert coding.reason == "self_coding_engine_failure"
    assert coding.should_exit is True

    cycle = classify_runtime_failure(event="self-improvement cycle thread crashed")
    assert cycle.category == "critical"
    assert cycle.reason == "self_improvement_cycle_failure"
    assert cycle.should_exit is True


def test_classify_runtime_failure_non_critical_buckets():
    telemetry = classify_runtime_failure(component="synergy_exporter", event="telemetry write failure")
    assert telemetry.category == "non_critical"
    assert telemetry.reason == "telemetry_failure"
    assert telemetry.should_exit is False

    optional_dep = classify_runtime_failure(event="optional dependency missing")
    assert optional_dep.category == "non_critical"
    assert optional_dep.reason == "optional_dependency_failure"
    assert optional_dep.should_exit is False

    api = classify_runtime_failure(error=RuntimeError("integration API timeout"))
    assert api.category == "non_critical"
    assert api.reason == "integration_or_api_error"
    assert api.should_exit is False
