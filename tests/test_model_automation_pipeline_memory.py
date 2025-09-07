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
import menace.memory_bot as mb  # noqa: E402
from menace.menace_memory_manager import MenaceMemoryManager  # noqa: E402
from menace.db_router import DBRouter  # noqa: E402
import types  # noqa: E402


class DummyAggregator:
    def __init__(self):
        self.info_db = rab.InfoDB(":memory:")

    def process(self, topic: str, energy: int = 1):
        return []


class DummyPathway:
    def similar_actions(self, action: str, limit: int = 1):
        return [("a", 2.0)]


def test_memory_search_called(monkeypatch, tmp_path):
    agg = DummyAggregator()
    mem_bot = mb.MemoryBot(mb.MemoryStorage(tmp_path / "m.json.gz"))
    db_router = DBRouter(memory_mgr=MenaceMemoryManager(tmp_path / "mem.db"))
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
        workflow_db=thb.WorkflowDB(tmp_path / "wf.db"),
        memory_bot=mem_bot,
        pathway_db=DummyPathway(),
        myelination_threshold=1.0,
        db_router=db_router,
        context_builder=builder,
    )
    monkeypatch.setattr(pipeline, "_gather_research", lambda model, energy: [])
    monkeypatch.setattr(pipeline, "_items_to_tasks", lambda items: [])
    monkeypatch.setattr(pipeline, "_plan_bots", lambda tasks: [])
    monkeypatch.setattr(pipeline, "_predict_resources", lambda plans: {})
    monkeypatch.setattr(pipeline, "_roi", lambda model, tasks: prb.ROIResult(1.0, 0, 0, 1.0, 0))
    monkeypatch.setattr(
        pipeline.roi_bot,
        "handoff_to_implementation",
        lambda b, opt, title: thb.TaskPackage(tasks=[]),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_support_bots",
        lambda model, *, energy=1.0, weight=1.0: None,
    )
    called = {}

    def fake_search(term, limit=5):
        called['hit'] = True
        return []
    monkeypatch.setattr(pipeline.memory_bot, "search", fake_search)
    monkeypatch.setattr(pipeline.db_router.memory_mgr, "search_by_tag", lambda t: [])
    pipeline.run("model")
    assert called.get('hit', False)
