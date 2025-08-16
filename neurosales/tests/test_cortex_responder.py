import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.cortex_responder import CortexAwareResponder, InMemoryResponseDB
from neurosales.memory import ConversationMemory
from neurosales.user_preferences import PreferenceProfile
from unittest.mock import patch


def test_cortex_pipeline_basic():
    memory = ConversationMemory()
    profile = PreferenceProfile(embedding=[0.0] * 384, archetype="helper")
    db = InMemoryResponseDB()
    with patch("neurosales.external_integrations.GPT4Client.stream_chat", return_value=["Sure."]):
        with patch("neurosales.external_integrations.PineconeLogger.log") as log, \
             patch("neurosales.embedding.embed_text", return_value=[0.0] * 384):
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
