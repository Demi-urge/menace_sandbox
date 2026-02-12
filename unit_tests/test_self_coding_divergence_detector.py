from self_coding_divergence_detector import (
    CycleMetricsRecord,
    DivergenceDetectorConfig,
    SelfCodingDivergenceDetector,
)


def _record(cycle: int, reward: float, *, revenue: float | None = None, profit: float | None = None) -> CycleMetricsRecord:
    return CycleMetricsRecord(
        cycle_index=cycle,
        bot_id="alpha",
        workflow_id="wf-alpha",
        reward_score=reward,
        revenue=revenue,
        profit=profit,
    )


def test_true_positive_divergence_reward_up_real_down():
    detector = SelfCodingDivergenceDetector(
        DivergenceDetectorConfig(window_size=3, flatness_threshold=0.0, minimum_effect_size=0.2)
    )
    history = [
        _record(1, 1.0, profit=10.0),
        _record(2, 1.4, profit=9.7),
        _record(3, 1.9, profit=9.0),
    ]

    result = detector.evaluate(history)

    assert result.triggered is True
    assert result.metric_name == "profit"
    assert result.reward_delta >= 0.2
    assert result.real_metric_delta <= 0.0


def test_no_trigger_when_real_metrics_improve_with_reward():
    detector = SelfCodingDivergenceDetector(
        DivergenceDetectorConfig(window_size=3, flatness_threshold=0.0, minimum_effect_size=0.2)
    )
    history = [
        _record(1, 1.0, revenue=10.0),
        _record(2, 1.4, revenue=10.6),
        _record(3, 1.7, revenue=11.2),
    ]

    result = detector.evaluate(history)

    assert result.triggered is False


def test_noisy_boundary_obeys_flatness_and_effect_size_thresholds():
    detector = SelfCodingDivergenceDetector(
        DivergenceDetectorConfig(window_size=4, flatness_threshold=-0.03, minimum_effect_size=0.25)
    )
    near_boundary = [
        _record(1, 1.00, revenue=10.00),
        _record(2, 1.06, revenue=10.02),
        _record(3, 1.15, revenue=9.99),
        _record(4, 1.24, revenue=9.98),
    ]
    should_trigger = near_boundary + [_record(5, 1.34, revenue=9.94)]

    first = detector.evaluate(near_boundary)
    second = detector.evaluate(should_trigger[-4:])

    assert first.triggered is False
    assert second.triggered is True
