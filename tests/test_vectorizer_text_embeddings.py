import numpy as np
import vector_service.vectorizer as vz


def test_encode_text_with_sentence_transformer(monkeypatch):
    """Text embeds using a provided SentenceTransformer."""

    class DummyST:
        def encode(self, texts):
            return np.array([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(vz, "SentenceTransformer", DummyST)
    captured = {}

    def fake_governed(text, embedder):
        captured["embedder"] = embedder
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(vz, "governed_embed", fake_governed)
    svc = vz.SharedVectorService(text_embedder=DummyST())
    vec = svc.vectorise("text", {"text": "hello"})
    assert vec == [1.0, 2.0, 3.0]
    assert captured.get("embedder") is svc.text_embedder


def test_encode_text_without_sentence_transformer(monkeypatch):
    """Fallback to a bundled model when SentenceTransformer is missing."""

    monkeypatch.setattr(vz, "SentenceTransformer", None)
    monkeypatch.setattr(vz, "_local_embed", lambda _: [0.1, 0.2])
    svc = vz.SharedVectorService(text_embedder=None)
    vec = svc.vectorise("text", {"text": "hello"})
    assert vec == [0.1, 0.2]


def test_encode_text_auto_initialises_embedder(monkeypatch):
    """Lazy embedder initialisation is attempted when missing."""

    class DummyEmbedder:
        def encode(self, texts):
            return np.array([[0.4, 0.5, 0.6]])

    embed_instance = DummyEmbedder()

    def fake_get_embedder():
        fake_get_embedder.called = True
        return embed_instance

    fake_get_embedder.called = False
    monkeypatch.setattr(vz, "SentenceTransformer", object)
    monkeypatch.setattr(vz, "get_embedder", fake_get_embedder)

    def fake_governed(text, embedder):
        fake_governed.used_embedder = embedder
        return [0.4, 0.5, 0.6]

    fake_governed.used_embedder = None
    monkeypatch.setattr(vz, "governed_embed", fake_governed)

    svc = vz.SharedVectorService(text_embedder=None)
    vec = svc.vectorise("text", {"text": "hello"})

    assert vec == [0.4, 0.5, 0.6]
    assert fake_get_embedder.called is True
    assert fake_governed.used_embedder is embed_instance
    assert svc.text_embedder is embed_instance
