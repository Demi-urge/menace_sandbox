
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
