import sandbox_runner.environment as env


def test_scenario_specific_metrics():
    metrics = {"error_rate": 0.1, "failure_count": 2, "throughput": 5.0}
    lat = env._scenario_specific_metrics("high_latency_api", metrics)
    hos = env._scenario_specific_metrics("hostile_input", metrics)
    mis = env._scenario_specific_metrics("user_misuse", metrics)
    con = env._scenario_specific_metrics("concurrency_spike", metrics)
    assert lat["latency_error_rate"] == 0.1
    assert hos["hostile_failures"] == 2
    assert mis["misuse_failures"] == 2
    assert con["concurrency_throughput"] == 5.0
