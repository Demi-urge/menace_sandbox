import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner


def test_embeddings_persist_without_indices(monkeypatch):
    tokens = []

    def fake_embed(text, embedder=None):
        tokens.append(text)
        return [3.0, 4.0]

    class DummyEmbedder:
        def get_sentence_embedding_dimension(self):
            return 2

    persisted = {}

    def fake_persist(kind, record_id, vec, *_, **__):
        persisted['kind'] = kind
        persisted['id'] = record_id
        persisted['vec'] = list(vec)

    monkeypatch.setattr(mwp, 'governed_embed', fake_embed)
    monkeypatch.setattr(mwp, 'get_embedder', lambda: DummyEmbedder())
    monkeypatch.setattr(mwp, 'persist_embedding', fake_persist)

    planner = MetaWorkflowPlanner()
    workflow = {"workflow": [{"function": "f1", "module": "m1", "tags": ["t1"]}]}
    vec = planner.encode('wf1', workflow)

    assert persisted['kind'] == 'workflow_meta'
    assert persisted['id'] == 'wf1'
    assert persisted['vec'] == vec

    assert set(tokens) == {"f1", "m1", "t1"}
    base = 2 + planner.roi_window + 2 + planner.roi_window
    assert vec[base:base+2] == [0.6, 0.8]
    assert vec[base+2:base+4] == [0.6, 0.8]
    assert vec[base+4:base+6] == [0.6, 0.8]
    assert not hasattr(planner, 'function_index')
    assert not hasattr(planner, 'module_index')
    assert not hasattr(planner, 'tag_index')
