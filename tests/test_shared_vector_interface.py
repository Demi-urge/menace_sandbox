import types
from vector_service import SharedVectorService
from bot_vectorizer import BotVectorizer


class DummyStore:
    def __init__(self):
        self.calls = []

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.calls.append((kind, record_id, list(vector)))

    def query(self, vector, top_k=5):  # pragma: no cover - not needed in tests
        return []

    def load(self):  # pragma: no cover - not needed in tests
        pass


def test_shared_vector_service_dispatch_and_store():
    store = DummyStore()

    class DummyEmbedder:
        class _Vec(list):
            def tolist(self):
                return list(self)

        def encode(self, texts):
            return [self._Vec([1.0, 0.5]) for _ in texts]

    svc = SharedVectorService(DummyEmbedder(), vector_store=store)

    svc.vectorise_and_store('action','a1',{'action_type':'run','domain':'test'})
    svc.vectorise_and_store('error','e1',{'category':'bug','root_module':'mod','stack_trace':'x'})
    svc.vectorise_and_store('workflow','w1',{'category':'ops','status':'new','workflow':[]})
    svc.vectorise_and_store('enhancement','h1',{'type':'ui','category':'ux','tags':[]})
    svc.vectorise_and_store('bot','b1',{'type':'utility','status':'active','tasks':[]})
    vec = svc.vectorise_and_store('text','t1',{'text':'hello'})

    assert vec == [1.0, 0.5]
    kinds = [c[0] for c in store.calls]
    assert kinds == ['action','error','workflow','enhancement','bot','text']


def test_gpt_memory_routes_through_vector_service():
    store = DummyStore()

    class DummyEmbedder:
        tokenizer = types.SimpleNamespace(encode=lambda s: [0])

        class _Vec(list):
            def tolist(self):
                return list(self)

        def encode(self, texts):
            return [self._Vec([0.25]) for _ in texts]

    svc = SharedVectorService(DummyEmbedder(), vector_store=store)

    # ``gpt_memory`` depends on modules with package-relative imports which are
    # not available in the test environment.  Provide lightweight stubs so the
    # import succeeds without pulling in the heavy dependencies.
    import sys
    sys.modules.setdefault(
        "data_bot",
        types.SimpleNamespace(MetricsDB=object),
    )
    sys.modules.setdefault(
        "scope_utils",
        types.SimpleNamespace(
            Scope=object,
            build_scope_clause=lambda *a, **k: "",
            apply_scope=lambda *a, **k: None,
        ),
    )

    from menace_sandbox.gpt_memory import GPTMemoryManager
    mem = GPTMemoryManager(db_path=':memory:', vector_service=svc)
    mem.log_interaction('hi','there')
    mem.close()

    assert store.calls and store.calls[0][0] == 'text'


def test_shared_vector_service_bot_embedding_dim():
    svc = SharedVectorService()
    bot = {"type": "assistant", "status": "active", "tasks": ["t"], "estimated_profit": 0}
    vec = svc.vectorise("bot", bot)
    assert len(vec) == BotVectorizer().dim
