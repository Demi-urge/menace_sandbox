import types
from vector_service import SharedVectorService


def test_shared_vector_service_dispatch_and_store(monkeypatch):
    calls = []
    def fake_persist(kind, record_id, vec, *, path="embeddings.jsonl"):
        calls.append((kind, record_id, list(vec), path))
    monkeypatch.setattr('vector_service.vectorizer.persist_embedding', fake_persist)

    class DummyEmbedder:
        def encode(self, text):
            return [1.0, 0.5]
    svc = SharedVectorService(DummyEmbedder())

    svc.vectorise_and_store('action','a1',{'action_type':'run','domain':'test'})
    svc.vectorise_and_store('error','e1',{'category':'bug','root_module':'mod','stack_trace':'x'})
    svc.vectorise_and_store('workflow','w1',{'category':'ops','status':'new','workflow':[]})
    svc.vectorise_and_store('enhancement','h1',{'type':'ui','category':'ux','tags':[]})
    vec = svc.vectorise_and_store('text','t1',{'text':'hello'})

    assert vec == [1.0, 0.5]
    kinds = [c[0] for c in calls]
    assert kinds == ['action','error','workflow','enhancement','text']


def test_gpt_memory_routes_through_vector_service(monkeypatch):
    calls = []
    def fake_persist(kind, record_id, vec, *, path='embeddings.jsonl'):
        calls.append((kind, record_id, list(vec), path))
    monkeypatch.setattr('vector_service.vectorizer.persist_embedding', fake_persist)

    class DummyEmbedder:
        tokenizer = types.SimpleNamespace(encode=lambda s: [0])
        def encode(self, text):
            return [0.25]

    svc = SharedVectorService(DummyEmbedder())
    from gpt_memory import GPTMemoryManager
    mem = GPTMemoryManager(db_path=':memory:', vector_service=svc)
    mem.log_interaction('hi','there')
    mem.close()

    assert calls and calls[0][0] == 'text'
