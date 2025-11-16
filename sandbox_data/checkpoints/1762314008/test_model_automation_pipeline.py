import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.model_automation_pipeline as mapl  # noqa: E402
import menace.research_aggregator_bot as rab  # noqa: E402
import menace.task_handoff_bot as thb  # noqa: E402
import menace.information_synthesis_bot as isb  # noqa: E402
import menace.task_validation_bot as tvb  # noqa: E402
import menace.bot_planning_bot as bpb  # noqa: E402
import menace.pre_execution_roi_bot as prb  # noqa: E402
import menace.implementation_optimiser_bot as iob  # noqa: E402
import menace.resource_prediction_bot as rpb  # noqa: E402
import types  # noqa: E402


class DummyAggregator:
    def __init__(self):
        self.info_db = rab.InfoDB(":memory:")

    def process(self, topic: str, energy: int = 1):
        return [
            rab.ResearchItem(
                topic=topic,
                content="data",
                timestamp=0.0,
                title=topic,
                tags=["workflow"],
                category="workflow",
            )
        ]


def test_builder_required():
    with pytest.raises(TypeError):
        mapl.ModelAutomationPipeline()
    with pytest.raises(ValueError):
        mapl.ModelAutomationPipeline(context_builder=None)


def test_pipeline_runs(tmp_path):
    agg = DummyAggregator()
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    pipeline = mapl.ModelAutomationPipeline(
        aggregator=agg,
        synthesis_bot=isb.InformationSynthesisBot(
            db_url="sqlite:///:memory:", aggregator=agg, context_builder=builder
        ),
        validator=tvb.TaskValidationBot(["model"]),
        planner=bpb.BotPlanningBot(),
        hierarchy=mapl.HierarchyAssessmentBot(),
        predictor=rpb.ResourcePredictionBot(),
        roi_bot=prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv")),
        handoff=thb.TaskHandoffBot(api_url="http://localhost"),
        optimiser=iob.ImplementationOptimiserBot(context_builder=builder),
        workflow_db=thb.WorkflowDB(tmp_path / "wf.db"),
        funds=100.0,
        context_builder=builder,
    )
    result = pipeline.run("model")
    assert result.package is not None
    assert result.roi is not None


def test_reuse_granular(monkeypatch, tmp_path):
    agg = DummyAggregator()
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["step1", "step2"], title="model", description="model"))
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    pipeline = mapl.ModelAutomationPipeline(
        aggregator=agg,
        synthesis_bot=isb.InformationSynthesisBot(
            db_url="sqlite:///:memory:", aggregator=agg, context_builder=builder
        ),
        validator=tvb.TaskValidationBot(["model"]),
        planner=bpb.BotPlanningBot(),
        hierarchy=mapl.HierarchyAssessmentBot(),
        predictor=rpb.ResourcePredictionBot(),
        roi_bot=prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv")),
        handoff=thb.TaskHandoffBot(),
        optimiser=iob.ImplementationOptimiserBot(context_builder=builder),
        workflow_db=wf_db,
        funds=0.0,
        context_builder=builder,
    )

    monkeypatch.setattr(
        pipeline,
        "_items_to_tasks",
        lambda items: [
            isb.SynthesisTask(description="base", urgency=1, complexity=1, category="analysis")
        ],
    )
    captured = {}

    def capture(tasks):
        captured["desc"] = [t.description for t in tasks]
        return list(tasks)

    monkeypatch.setattr(pipeline, "_validate_tasks", capture)
    monkeypatch.setattr(pipeline, "_plan_bots", lambda tasks: [])
    monkeypatch.setattr(pipeline, "_predict_resources", lambda plans: {})
    monkeypatch.setattr(
        pipeline,
        "_roi",
        lambda model, tasks: prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
    )
    monkeypatch.setattr(
        pipeline.roi_bot,
        "handoff_to_implementation",
        lambda build_tasks, opt, title: thb.TaskPackage(tasks=[]),
    )
    monkeypatch.setattr(
        pipeline, "_run_support_bots", lambda model, *, energy=1.0, weight=1.0: None
    )

    pipeline.run("model")
    descs = captured.get("desc", [])
    assert "Reuse workflow step1" in descs
    assert "Reuse workflow step2" in descs
