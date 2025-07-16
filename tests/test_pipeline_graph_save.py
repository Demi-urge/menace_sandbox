import pytest
import pytest

pytest.importorskip("networkx")

import menace.model_automation_pipeline as mapl
import menace.research_aggregator_bot as rab
import menace.task_handoff_bot as thb
import menace.information_synthesis_bot as isb
import menace.task_validation_bot as tvb
import menace.bot_planning_bot as bpb
import menace.pre_execution_roi_bot as prb
import menace.implementation_optimiser_bot as iob
import menace.resource_prediction_bot as rpb


class DummyAggregator:
    def __init__(self):
        self.info_db = rab.InfoDB(":memory:")
    def process(self, topic: str, energy: int = 1):
        return []


def test_pipeline_calls_save(monkeypatch, tmp_path):
    agg = DummyAggregator()
    pipeline = mapl.ModelAutomationPipeline(
        aggregator=agg,
        synthesis_bot=isb.InformationSynthesisBot(db_url="sqlite:///:memory:"),
        validator=tvb.TaskValidationBot(["model"]),
        planner=bpb.BotPlanningBot(),
        hierarchy=mapl.HierarchyAssessmentBot(),
        predictor=rpb.ResourcePredictionBot(),
        roi_bot=prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv")),
        handoff=thb.TaskHandoffBot(),
        optimiser=iob.ImplementationOptimiserBot(),
        workflow_db=thb.WorkflowDB(tmp_path / "wf.db"),
        pathway_db=mapl.PathwayDB(tmp_path / "p.db"),
    )

    monkeypatch.setattr(pipeline, "_gather_research", lambda m, energy: [])
    monkeypatch.setattr(pipeline, "_items_to_tasks", lambda items: [])
    monkeypatch.setattr(pipeline, "_plan_bots", lambda tasks: [])
    monkeypatch.setattr(pipeline, "_validate_plan", lambda plans: [])
    monkeypatch.setattr(pipeline, "_predict_resources", lambda plans: {})
    monkeypatch.setattr(pipeline, "_roi", lambda model, tasks: prb.ROIResult(0,0,0,0,0))
    monkeypatch.setattr(pipeline.roi_bot, "handoff_to_implementation", lambda bt,opt,title: None)
    monkeypatch.setattr(pipeline, "_run_support_bots", lambda model, *, energy=1.0, weight=1.0: None)

    called = {}
    def fake_save(dest):
        called["dest"] = dest
    monkeypatch.setattr(pipeline.bot_registry, "save", fake_save)

    pipeline.run("model")
    assert called.get("dest") is pipeline.pathway_db
