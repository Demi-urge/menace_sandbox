import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.memory import ConversationMemory
from unittest.mock import patch
import types


def test_cortex_pipeline_basic(monkeypatch):
    class DummyCB:
        def build(self, query: str) -> str:
            return ""

        def refresh_db_weights(self):
            return None

    dummy_mod = types.SimpleNamespace(ContextBuilder=DummyCB)
    monkeypatch.setitem(sys.modules, "vector_service", types.SimpleNamespace(context_builder=dummy_mod))
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", dummy_mod)
    monkeypatch.setitem(
        sys.modules, "neurosales.embedding", types.SimpleNamespace(embed_text=lambda *a, **k: [0.0] * 384)
    )

    from neurosales.cortex_responder import CortexAwareResponder, InMemoryResponseDB
    class DummyProfile:
        def __init__(self, embedding, archetype):
            self.embedding = embedding
            self.archetype = archetype

    memory = ConversationMemory()
    profile = DummyProfile(embedding=[0.0] * 384, archetype="helper")
    db = InMemoryResponseDB()
    with patch("neurosales.external_integrations.GPT4Client.stream_chat", return_value=["Sure."]):
        with patch("neurosales.external_integrations.PineconeLogger.log") as log:
            responder = CortexAwareResponder(
                "k",
                pinecone_index="idx",
                pinecone_key="k",
                pinecone_env="us-east",
                pg=db,
            )
            out = responder.generate_response("s1", "u1", "hello", memory, profile)
    assert out
    assert db.rows
    assert log.called
    assert responder.queue.heap
