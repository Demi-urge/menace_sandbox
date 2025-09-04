import sys
import types
import importlib.util
from pathlib import Path
from dynamic_path_router import resolve_path


def _load_wcs(monkeypatch):
    vector_utils = types.ModuleType("vector_utils")
    vector_utils.cosine_similarity = (
        lambda a, b: sum(x * y for x, y in zip(a, b))
        / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) or 1.0)
    )
    monkeypatch.setitem(sys.modules, "vector_utils", vector_utils)

    class DummyROIResultsDB:
        def fetch_trends(self, wid):
            return []

    roi_mod = types.ModuleType("roi_results_db")
    roi_mod.ROIResultsDB = DummyROIResultsDB
    monkeypatch.setitem(sys.modules, "roi_results_db", roi_mod)

    class DummyStabilityDB:
        def is_stable(self, wid):
            return False

        def get_ema(self, wid):
            return 0.0, None

    stab_mod = types.ModuleType("workflow_stability_db")
    stab_mod.WorkflowStabilityDB = DummyStabilityDB
    monkeypatch.setitem(sys.modules, "workflow_stability_db", stab_mod)

    class DummyWSC:
        @staticmethod
        def _entropy(spec):
            return 0.0

    wsc_mod = types.ModuleType("workflow_synergy_comparator")
    wsc_mod.WorkflowSynergyComparator = DummyWSC
    monkeypatch.setitem(sys.modules, "workflow_synergy_comparator", wsc_mod)

    monkeypatch.setitem(sys.modules, "task_handoff_bot", types.ModuleType("task_handoff_bot"))

    spec = importlib.util.spec_from_file_location(
        "workflow_chain_suggester",
        resolve_path("workflow_chain_suggester.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_chain_suggester"] = module
    spec.loader.exec_module(module)
    return module


def test_kmeans_clusters_vectors(monkeypatch):
    wcs = _load_wcs(monkeypatch)
    sugg = wcs.WorkflowChainSuggester(wf_db=None, roi_db=None, stability_db=None)
    vectors = [
        ("a", [0.0, 0.1]),
        ("b", [0.0, 1.0]),
        ("c", [10.0, 10.0]),
        ("d", [10.0, 11.0]),
    ]
    monkeypatch.setattr(wcs.random, "sample", lambda seq, k: [seq[0], seq[2]])
    clusters = sugg._kmeans(vectors, k=2, iterations=5)
    cluster_sets = {frozenset(c) for c in clusters}
    assert cluster_sets == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_suggest_chains_roi_weighted_selection(monkeypatch):
    wcs = _load_wcs(monkeypatch)

    class DummyDB:
        def search_by_vector(self, vec, top_k):
            return [("1", 0.0), ("2", 0.0)]

        def get_vector(self, wid):
            return [1.0, 0.0]

    sugg = wcs.WorkflowChainSuggester(wf_db=DummyDB(), roi_db=None, stability_db=None)
    monkeypatch.setattr(sugg, "_roi_score", lambda wid: 1.0 if wid == "1" else 0.0)
    monkeypatch.setattr(sugg, "_roi_delta", lambda wid: 0.0)
    monkeypatch.setattr(sugg, "_stability_weight", lambda wid: 1.0)
    monkeypatch.setattr(sugg, "_kmeans", lambda vectors, k: [[wid] for wid, _ in vectors])

    chains = sugg.suggest_chains([0.0, 0.0], top_k=2)
    assert chains == [["1"], ["2"]]


def test_mutation_helpers(monkeypatch):
    wcs = _load_wcs(monkeypatch)
    chain = ["a", "b", "c"]
    assert wcs.WorkflowChainSuggester.swap_steps(chain, 0, 2) == ["c", "b", "a"]
    assert wcs.WorkflowChainSuggester.split_sequence(chain, 1) == [["a"], ["b", "c"]]
    merged = wcs.WorkflowChainSuggester.merge_partial_chains([["a", "b"], ["b", "c"]])
    assert merged == ["a", "b", "c"]

