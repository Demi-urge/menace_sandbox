import numpy as np
import vector_service.vectorizer as vz


class DummyEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 2.0, 3.0]])


def test_encode_text_with_sentence_transformer():
    svc = vz.SharedVectorService(text_embedder=DummyEmbedder())
    vec = svc.vectorise("text", {"text": "hello"})
    assert vec == [1.0, 2.0, 3.0]


def test_encode_text_without_sentence_transformer(monkeypatch):
    monkeypatch.setattr(vz, "SentenceTransformer", None)
    svc = vz.SharedVectorService(text_embedder=None)
    vec = svc.vectorise("text", {"text": "hello"})
    assert isinstance(vec, list) and len(vec) > 0
