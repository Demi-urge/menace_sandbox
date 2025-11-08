import os
import sys
import numpy as np
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from neurosales import embedding_memory as emem


def _disable_models(monkeypatch):
    monkeypatch.setattr(emem, "SentenceTransformer", None)
    monkeypatch.setattr(emem, "faiss", None)
    monkeypatch.setattr(emem, "np", None)
    from compliance.license_fingerprint import check as lc
    from security.secret_redactor import redact as red
    from analysis.semantic_diff_filter import find_semantic_risks as fsr
    monkeypatch.setattr(emem, "license_check", lc)
    monkeypatch.setattr(emem, "redact", red)
    monkeypatch.setattr(emem, "find_semantic_risks", fsr)


def test_basic_storage(monkeypatch):
    _disable_models(monkeypatch)
    mem = emem.EmbeddingConversationMemory(max_messages=3)
    mem.add_message('user', 'hello world')
    mem.add_message('assistant', 'hi there')
    mem.add_message('user', 'another message')
    recent = mem.get_recent_messages()
    assert len(recent) == 3
    assert recent[0].content == 'hello world'
    sim = mem.most_similar('hello')
    assert isinstance(sim, list)


def test_license_skip(monkeypatch):
    _disable_models(monkeypatch)
    mem = emem.EmbeddingConversationMemory(max_messages=3)
    mem.add_message('user', 'gnu general public license')
    assert len(mem.get_recent_messages()) == 0


def test_semantic_alerts(monkeypatch):
    class DummyModel:
        def __init__(self, name: str, *args, **kwargs) -> None:
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, text: str, convert_to_numpy: bool = True):
            return np.array([float(len(text)), 0.0], dtype="float32")

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            pass

        def reset(self) -> None:
            pass

        def add(self, vec) -> None:
            pass

        def search(self, query, k):
            return np.array([[0.0]]), np.array([[0]])

    dummy_faiss = types.SimpleNamespace(IndexFlatL2=DummyIndex)
    monkeypatch.setattr(emem, "SentenceTransformer", DummyModel)
    monkeypatch.setattr(emem, "faiss", dummy_faiss)
    monkeypatch.setattr(emem, "np", np)
    from compliance.license_fingerprint import check as lc
    from security.secret_redactor import redact as red
    from analysis.semantic_diff_filter import find_semantic_risks as fsr
    monkeypatch.setattr(emem, "license_check", lc)
    monkeypatch.setattr(emem, "redact", red)
    monkeypatch.setattr(emem, "find_semantic_risks", fsr)
    mem = emem.EmbeddingConversationMemory(max_messages=3)
    mem.add_message('user', "eval('data')")
    res = mem.most_similar('eval')
    assert res and res[0].alerts
