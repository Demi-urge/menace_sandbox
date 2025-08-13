import menace.human_alignment_flagger as haf
from sandbox_settings import SandboxSettings


def test_metric_improvement_still_warns(tmp_path):
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("tests: 1\ncomplexity: 1\n")
    settings = SandboxSettings(alignment_baseline_metrics_path=str(baseline))

    # Improved accuracy metric but reduced tests and higher complexity
    metrics = {"accuracy": 0.95, "previous_accuracy": 0.9}
    code = "def f(x):\n    if x:\n        return x\n"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": "module.py", "code": code}],
        metrics=metrics,
        logs=[],
        settings=settings,
    )
    assert metrics["accuracy"] > metrics["previous_accuracy"]
    issues = [w.get("issue", "") for w in warnings["maintainability"]]
    assert any("test count decreased" in issue for issue in issues)
    assert any("complexity increased" in issue for issue in issues)
