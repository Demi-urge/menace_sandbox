import sandbox_runner.environment as env


def test_scenario_specific_metrics():
    metrics = {
        "error_rate": 0.1,
        "failure_count": 2,
        "throughput": 5.0,
        "concurrency_threads": 8.0,
        "concurrency_tasks": 12.0,
        "concurrency_error_rate": 0.25,
        "concurrency_level": 4.0,
        "invalid_call_count": 3.0,
        "recovery_attempts": 1.0,
        "sanitization_failures": 2.0,
        "validation_failures": 1.0,
    }
    lat = env._scenario_specific_metrics("high_latency_api", metrics)
    hos = env._scenario_specific_metrics("hostile_input", metrics)
    mis = env._scenario_specific_metrics("user_misuse", metrics)
    con = env._scenario_specific_metrics("concurrency_spike", metrics)
    assert lat["latency_error_rate"] == 0.1
    assert hos["hostile_failures"] == 2
    assert hos["hostile_sanitization_failures"] == 2.0
    assert hos["hostile_validation_failures"] == 1.0
    assert mis["misuse_failures"] == 2
    assert mis["misuse_invalid_calls"] == 3.0
    assert mis["misuse_recovery_attempts"] == 1.0
    assert con["concurrency_throughput"] == 5.0
    assert con["concurrency_thread_saturation"] == 8.0
    assert con["concurrency_async_saturation"] == 12.0
    assert con["concurrency_error_count"] == 1.0
