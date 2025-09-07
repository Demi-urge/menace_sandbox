import os
import pytest
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
pytest.importorskip("networkx")
import types
import sys
from pathlib import Path

vs = types.ModuleType("vector_service")
class DummyBuilder:
    def __init__(self, *a, **k):
        pass
    def refresh_db_weights(self):
        pass
vs.ContextBuilder = DummyBuilder
vs.CognitionLayer = object
sys.modules["vector_service"] = vs

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules["menace"] = menace_pkg
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
        return []


class HighPathway:
    def similar_actions(self, action: str, limit: int = 1):
        return [("x", 2.5)]


class LowPathway:
    def similar_actions(self, action: str, limit: int = 1):
        return [("x", 0.5)]


def _setup_pipeline(pathway):
    agg = DummyAggregator()
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    return mapl.ModelAutomationPipeline(
        aggregator=agg,
        synthesis_bot=isb.InformationSynthesisBot(
            db_url="sqlite:///:memory:", aggregator=agg
        ),
        validator=tvb.TaskValidationBot(["model"]),
        planner=bpb.BotPlanningBot(),
        hierarchy=mapl.HierarchyAssessmentBot(),
        predictor=rpb.ResourcePredictionBot(),
        roi_bot=prb.PreExecutionROIBot(prb.ROIHistoryDB(":memory:")),
        handoff=thb.TaskHandoffBot(),
        optimiser=iob.ImplementationOptimiserBot(context_builder=builder),
        workflow_db=thb.WorkflowDB(":memory:"),
        pathway_db=pathway,
        myelination_threshold=1.0,
        context_builder=builder,
    )


def _stub_pipeline(pipe, monkeypatch, capture):
    monkeypatch.setattr(pipe, "_gather_research", lambda model, energy: [])
    monkeypatch.setattr(
        pipe,
        "_items_to_tasks",
        lambda items: [
            isb.SynthesisTask(description="task", urgency=1, complexity=1, category="a")
        ],
    )
    monkeypatch.setattr(pipe, "_validate_tasks", lambda tasks: list(tasks))

    def fake_plan(tasks, *, trust_weight=1.0):
        capture["weight"] = trust_weight
        n = int(round(trust_weight))
        capture["count"] = n
        return [
            bpb.BotPlan(name=f"b{i}", template="t", scalability=1.0, level="L1")
            for i in range(n)
        ]

    monkeypatch.setattr(pipe.planner, "plan_bots", fake_plan)
    monkeypatch.setattr(pipe, "_validate_plan", lambda plans: plans)
    monkeypatch.setattr(pipe, "_assess_hierarchy", lambda plans: None)
    monkeypatch.setattr(pipe, "_predict_resources", lambda plans: {})
    monkeypatch.setattr(pipe, "_roi", lambda model, tasks: prb.ROIResult(1.0, 0, 0, 1.0, 0))
    monkeypatch.setattr(
        pipe.roi_bot,
        "handoff_to_implementation",
        lambda bt, opt, title: thb.TaskPackage(tasks=[]),
    )
    monkeypatch.setattr(
        pipe,
        "_run_support_bots",
        lambda model, *, energy=1.0, weight=1.0: None,
    )


def test_trust_weight_increases_plans(monkeypatch):
    high_capture = {}
    pipe_high = _setup_pipeline(HighPathway())
    _stub_pipeline(pipe_high, monkeypatch, high_capture)
    pipe_high.run("m")

    low_capture = {}
    pipe_low = _setup_pipeline(LowPathway())
    _stub_pipeline(pipe_low, monkeypatch, low_capture)
    pipe_low.run("m")

    assert high_capture.get("weight", 0) > low_capture.get("weight", 0)
    assert high_capture.get("count", 0) > low_capture.get("count", 0)
