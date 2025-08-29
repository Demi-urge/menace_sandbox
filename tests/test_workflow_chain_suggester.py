
import logging

from workflow_chain_suggester import WorkflowChainSuggester


class DummyDB:
    def search_by_vector(self, vector, top_k):
        return [("1", 0.1), ("2", 0.1), ("3", 0.1)]

    def get_vector(self, rid):
        vecs = {
            "1": [1.0, 0.0, 0.0],
            "2": [0.0, 1.0, 0.0],
            "3": [0.0, 0.0, 1.0],
        }
        return vecs.get(str(rid))


class DummyROIDB:
    def fetch_trends(self, workflow_id):
        data = {
            "1": [{"roi_gain": 1.0}],
            "2": [{"roi_gain": 0.0}],
            "3": [{"roi_gain": 0.0}],
        }
        return data.get(str(workflow_id), [])


class DummyStabilityDB:
    def is_stable(self, workflow_id, current_roi=None, threshold=None):
        return str(workflow_id) != "2"

    def get_ema(self, workflow_id):
        return 0.0, 0


def test_suggest_chains_ranks_by_roi_and_stability():
    suggester = WorkflowChainSuggester(
        wf_db=DummyDB(), roi_db=DummyROIDB(), stability_db=DummyStabilityDB()
    )
    chains = suggester.suggest_chains([1.0, 0.0, 0.0], top_k=2)
    assert chains[0] == ["1"]
    assert chains[1] == ["3"]


def test_persisted_chain_is_prioritized(tmp_path, monkeypatch):
    from vector_utils import persist_embedding
    import workflow_chain_suggester as wcs

    path = tmp_path / "embeddings.jsonl"
    persist_embedding(
        "workflow_chain",
        "1->2",
        [1.0, 0.0, 0.0],
        path=path,
        metadata={"roi": 1.0, "entropy": 0.0},
    )
    orig = wcs._load_chain_embeddings
    monkeypatch.setattr(wcs, "_load_chain_embeddings", lambda path=path: orig(path))

    suggester = WorkflowChainSuggester(
        wf_db=DummyDB(), roi_db=DummyROIDB(), stability_db=DummyStabilityDB()
    )
    chains = suggester.suggest_chains([1.0, 0.0, 0.0], top_k=2)
    assert chains[0] == ["1", "2"]


def test_chain_mutation_helpers():
    chain = ["a", "b", "c"]
    assert WorkflowChainSuggester.swap_steps(chain, 0, 2) == ["c", "b", "a"]
    assert WorkflowChainSuggester.split_sequence(chain, 1) == [["a"], ["b", "c"]]
    merged = WorkflowChainSuggester.merge_partial_chains([["a", "b"], ["b", "c"]])
    assert merged == ["a", "b", "c"]


def test_high_entropy_chain_is_filtered(monkeypatch):
    import workflow_chain_suggester as wcs

    monkeypatch.setattr(
        wcs,
        "_load_chain_embeddings",
        lambda path=None: [
            {"id": "1->2", "vector": [1.0, 0.0, 0.0], "roi": 0.0, "entropy": 0.0}
        ],
    )

    def fake_entropy(cls, spec):
        mods = [s.get("module") for s in spec.get("steps", [])]
        return 2.0 if mods == ["1", "2"] else 0.0

    monkeypatch.setattr(
        wcs.WorkflowSynergyComparator,
        "_entropy",
        classmethod(fake_entropy),
    )

    suggester = WorkflowChainSuggester(
        wf_db=DummyDB(), roi_db=DummyROIDB(), stability_db=DummyStabilityDB()
    )
    chains = suggester.suggest_chains([1.0, 0.0, 0.0], top_k=2)
    assert ["1", "2"] not in chains


class BoomROI:
    """ROI DB stub that raises to trigger error handling."""

    def fetch_trends(self, workflow_id):  # pragma: no cover - simple stub
        raise RuntimeError("boom")


def test_roi_delta_logs_and_defaults(caplog):
    suggester = WorkflowChainSuggester(
        wf_db=DummyDB(), roi_db=BoomROI(), stability_db=DummyStabilityDB(),
    )

    caplog.set_level(logging.WARNING)
    caplog.clear()
    result = suggester._roi_delta("x")

    assert isinstance(result, float)
    assert result == 0.0
    assert any("ROI delta fetch failed" in rec.message for rec in caplog.records)
