import types
import sys

import networkx as nx
import pytest

import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner, simulate_meta_workflow


class DummyGraph:
    """Minimal graph wrapper returning empty I/O signatures."""

    def __init__(self, g: nx.DiGraph) -> None:
        self.graph = g

    def get_io_signature(self, _wid):
        return {"inputs": {}, "outputs": {}}


class DummyROI:
    def fetch_trends(self, workflow_id: str):
        return []


class DummyRunner:
    """Execute each workflow and capture the ROI result."""

    class ModuleMetric:
        def __init__(self, name: str, result: float, success: bool = True) -> None:
            self.name = name
            self.result = result
            self.success = success
            self.duration = 0.0

    class Metrics:
        def __init__(self, result: float) -> None:
            self.modules = [DummyRunner.ModuleMetric("m", result)]
            self.crash_count = 0

    def run(self, funcs):
        result = funcs[0]()
        return DummyRunner.Metrics(float(result))


class DummyRetriever:
    """Return fixed retrieval hits ensuring cross-domain candidate presence."""

    def _get_retriever(self):
        return self

    def retrieve(self, query_vec, top_k, dbs=None):
        hits = [
            types.SimpleNamespace(record_id="rd", score=0.9, metadata={"id": "rd"}),
            types.SimpleNamespace(record_id="em", score=0.8, metadata={"id": "em"}),
        ]
        return hits[:top_k], None, None


class DummyBuilder:
    def build(self, *_, **__):
        return {}

    def refresh_db_weights(self) -> None:
        pass


class WeightedComparator:
    """Synergy comparator favouring cross-domain pairs and providing entropy."""

    @staticmethod
    def compare(a, b):
        doms = {a.get("domain"), b.get("domain")}
        if doms == {"youtube", "reddit"}:
            score = 0.9
        elif doms == {"reddit", "email"}:
            score = 0.8
        else:
            score = 0.1
        return types.SimpleNamespace(aggregate=score)

    @staticmethod
    def _entropy(_spec):
        return 0.5


def test_cross_domain_pipeline_with_validation(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", WeightedComparator)
    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=WeightedComparator),
    )

    embeddings = {
        "yt": [1.0, 0.0],
        "rd": [0.0, 1.0],
        "em": [0.6, 0.4],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(MetaWorkflowPlanner, "encode_workflow", fake_encode)

    planner = MetaWorkflowPlanner(
        context_builder=DummyBuilder(),
        graph=DummyGraph(nx.DiGraph()),
        roi_db=DummyROI(),
    )
    planner.domain_index.update({"youtube": 1, "reddit": 2, "email": 3})
    planner.cluster_map = {
        ("__domain_transitions__",): {
            (1, 2): {"count": 10, "delta_roi": 2.0},
            (2, 3): {"count": 8, "delta_roi": 1.5},
            (1, 3): {"count": 5, "delta_roi": -1.0},
        }
    }

    workflows = {
        "yt": {"domain": "youtube"},
        "rd": {"domain": "reddit"},
        "em": {"domain": "email"},
    }

    retr = DummyRetriever()
    pipeline_no_synergy = planner.compose_pipeline(
        "yt",
        workflows,
        length=3,
        synergy_weight=0.0,
        context_builder=planner.context_builder,
        retriever=retr,
    )
    assert pipeline_no_synergy[1] == "em"

    pipeline = planner.compose_pipeline(
        "yt",
        workflows,
        length=3,
        synergy_weight=1.0,
        context_builder=planner.context_builder,
        retriever=retr,
    )
    assert pipeline == ["yt", "em", "rd"]

    meta_spec = {"steps": [{"workflow_id": wid} for wid in pipeline]}
    funcs = {"yt": lambda: 1.0, "rd": lambda: 2.0, "em": lambda: 3.0}
    result = simulate_meta_workflow(meta_spec, workflows=funcs, runner=DummyRunner())

    assert result["roi_gain"] == pytest.approx(6.0)
    assert result["failures"] == 0
    assert result["entropy"] == pytest.approx(0.5)
