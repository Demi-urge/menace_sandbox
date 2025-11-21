import threading

import numpy as np
import pytest

import governed_embeddings
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


def test_disable_embedder_surfaces_in_vector_service(monkeypatch):
    """Disabled embedders surface via diagnostics and block embedding."""

    class DummyStore:
        def add(self, *args, **kwargs):
            return None

    class DummyEmbedder:
        def encode(self, texts):
            return np.array([[9.0, 9.0, 9.0]])

    dummy_embedder = DummyEmbedder()
    stop_event = threading.Event()

    monkeypatch.setattr(vz, "load_handlers", lambda: {})
    monkeypatch.setattr(vz, "SentenceTransformer", object)
    monkeypatch.setattr(vz, "get_embedder", governed_embeddings.get_embedder)
    monkeypatch.setattr(vz, "governed_embed", governed_embeddings.governed_embed)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_DISABLED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_STOP_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_DISABLE_REASON", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_DISABLE_CALLER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_DISABLE_TRIGGER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_STOP_REASON", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_STOP_CALLER", None)
    monkeypatch.setattr(governed_embeddings, "_embedder_lock", lambda: None)

    def fake_initialise_embedder_with_timeout(
        timeout_override=None,
        *,
        suppress_timeout_log=False,
        requester=None,
        stop_event=None,
    ):
        if governed_embeddings._embedder_disabled(stop_event):
            return None
        governed_embeddings._EMBEDDER = dummy_embedder
        governed_embeddings._EMBEDDER_INIT_EVENT.set()
        return None

    monkeypatch.setattr(
        governed_embeddings,
        "_initialise_embedder_with_timeout",
        fake_initialise_embedder_with_timeout,
    )

    svc = vz.SharedVectorService(text_embedder=None, vector_store=DummyStore())

    governed_embeddings.disable_embedder(reason="unit-test", stop_event=stop_event)
    diagnostics = governed_embeddings.embedder_diagnostics()
    assert diagnostics["embedder_disabled"] is True
    assert diagnostics["disable_reason"] == "unit-test"

    with pytest.raises(RuntimeError):
        svc.vectorise("text", {"text": "blocked"})

    governed_embeddings._EMBEDDER_DISABLED = False
    stop_event.clear()
    governed_embeddings._clear_embedder_disable_metadata()
    governed_embeddings._EMBEDDER_INIT_EVENT.clear()
    governed_embeddings._EMBEDDER = None

    vec = svc.vectorise("text", {"text": "restored"})
    assert vec == [9.0, 9.0, 9.0]
    post_diagnostics = governed_embeddings.embedder_diagnostics()
    assert post_diagnostics["embedder_disabled"] is False
    assert post_diagnostics["embedder_ready"] is True
