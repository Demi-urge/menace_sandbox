import types
from vector_service import SharedVectorService
from bot_vectorizer import BotVectorizer


def test_shared_vector_service_dispatch_and_store(monkeypatch):
    calls = []
    def fake_persist(kind, record_id, vec, *, path="embeddings.jsonl"):
        calls.append((kind, record_id, list(vec), path))
    monkeypatch.setattr('vector_service.vectorizer.persist_embedding', fake_persist)

    class DummyEmbedder:
        class _Vec(list):
            def tolist(self):
                return list(self)

        def encode(self, texts):
            return [self._Vec([1.0, 0.5]) for _ in texts]
    svc = SharedVectorService(DummyEmbedder())

    svc.vectorise_and_store('action','a1',{'action_type':'run','domain':'test'})
    svc.vectorise_and_store('error','e1',{'category':'bug','root_module':'mod','stack_trace':'x'})
    svc.vectorise_and_store('workflow','w1',{'category':'ops','status':'new','workflow':[]})
    svc.vectorise_and_store('enhancement','h1',{'type':'ui','category':'ux','tags':[]})
    svc.vectorise_and_store('bot','b1',{'type':'utility','status':'active','tasks':[]})
    vec = svc.vectorise_and_store('text','t1',{'text':'hello'})

    assert vec == [1.0, 0.5]
    kinds = [c[0] for c in calls]
    assert kinds == ['action','error','workflow','enhancement','bot','text']


def test_gpt_memory_routes_through_vector_service(monkeypatch):
    calls = []
    def fake_persist(kind, record_id, vec, *, path='embeddings.jsonl'):
        calls.append((kind, record_id, list(vec), path))
    monkeypatch.setattr('vector_service.vectorizer.persist_embedding', fake_persist)

    class DummyEmbedder:
        tokenizer = types.SimpleNamespace(encode=lambda s: [0])

        class _Vec(list):
            def tolist(self):
                return list(self)

        def encode(self, texts):
            return [self._Vec([0.25]) for _ in texts]

    svc = SharedVectorService(DummyEmbedder())
    from gpt_memory import GPTMemoryManager
    mem = GPTMemoryManager(db_path=':memory:', vector_service=svc)
    mem.log_interaction('hi','there')
    mem.close()

    assert calls and calls[0][0] == 'text'


def test_shared_vector_service_bot_embedding_dim():
    svc = SharedVectorService()
    bot = {"type": "assistant", "status": "active", "tasks": ["t"], "estimated_profit": 0}
    vec = svc.vectorise("bot", bot)
    assert len(vec) == BotVectorizer().dim
