import numpy as np
import vector_service.vectorizer as vz


def test_encode_text_with_sentence_transformer(monkeypatch):
    """Text embeds using a provided SentenceTransformer."""

    class DummyST:
        def encode(self, texts):
            return np.array([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(vz, "SentenceTransformer", DummyST)
    svc = vz.SharedVectorService(text_embedder=DummyST())
    vec = svc.vectorise("text", {"text": "hello"})
    assert vec == [1.0, 2.0, 3.0]


def test_encode_text_without_sentence_transformer(monkeypatch):
    """Fallback to a bundled model when SentenceTransformer is missing."""

    monkeypatch.setattr(vz, "SentenceTransformer", None)
    monkeypatch.setattr(vz, "_local_embed", lambda _: [0.1, 0.2])
    svc = vz.SharedVectorService(text_embedder=None)
    vec = svc.vectorise("text", {"text": "hello"})
    assert vec == [0.1, 0.2]
