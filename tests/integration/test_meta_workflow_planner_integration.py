import os
import networkx as nx

from meta_workflow_planner import MetaWorkflowPlanner
from roi_results_db import ROIResultsDB


class DummyGraph:
    def __init__(self, g: nx.DiGraph) -> None:
        self.graph = g


def test_small_chain_roi_improvement(tmp_path):
    db = ROIResultsDB(path=tmp_path / "roi.db")
    db.log_result(
        workflow_id="A",
        run_id="1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=-1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )
    db.log_result(
        workflow_id="A",
        run_id="2",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=3.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )
    db.log_result(
        workflow_id="B",
        run_id="1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=2.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )

    g = nx.DiGraph()
    g.add_edge("A", "B")
    planner = MetaWorkflowPlanner(graph=DummyGraph(g), roi_db=db, roi_window=5)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    vec_a = planner.encode("A", {"workflow": []})
    vec_b = planner.encode("B", {"workflow": []})
    os.chdir(old_cwd)

    roi_a = vec_a[2:7]
    assert roi_a[0] == -1.0 and roi_a[1] == 3.0
    assert vec_a[1] == 1.0  # branching
    roi_b = vec_b[2:7]
    assert roi_b[0] == 2.0
    assert vec_b[0] == 1.0  # depth
    assert roi_a[1] > roi_a[0]
