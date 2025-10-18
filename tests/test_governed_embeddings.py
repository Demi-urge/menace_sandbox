import inspect
import logging
import os
import sys
import threading
import types

import menace.governed_embeddings as governed_embeddings
import menace.local_knowledge_module as lkm
import menace.menace_memory_manager as mmm
from menace.governed_embeddings import governed_embed


def test_governed_embed_blocks_gpl(caplog):
    text = "This code is licensed under the GNU General Public License"
    with caplog.at_level(logging.WARNING):
        assert governed_embed(text) is None
    assert any("license" in r.msg for r in caplog.records)


def test_governed_embed_redacts_and_embeds(monkeypatch, caplog):
    recorded = {}

    def fake_redact(text: str) -> str:
        recorded["redacted"] = text
        return "clean"

    def fake_check(text: str):
        recorded["checked"] = text
        return None

    class DummyEmbedder:
        def encode(self, arr):
            assert arr[0] == "clean"
            return [types.SimpleNamespace(tolist=lambda: [0.1, 0.2])]

    monkeypatch.setattr(governed_embeddings, "redact", fake_redact)
    monkeypatch.setattr(governed_embeddings, "license_check", fake_check)
    monkeypatch.setattr(governed_embeddings, "get_embedder", lambda: DummyEmbedder())
    with caplog.at_level(logging.WARNING):
        vec = governed_embed("secret token")
    assert vec == [0.1, 0.2]
    assert recorded["redacted"] == "secret token"
    assert recorded["checked"] == "secret token"
    assert any("redacted" in r.msg for r in caplog.records)


def test_memory_manager_uses_governed_embed(monkeypatch):
    called = {}

    def fake_embed(text: str, embedder=None):
        called["text"] = text
        return [0.0]

    monkeypatch.setattr(mmm, "governed_embed", fake_embed)
    params = inspect.signature(mmm.MenaceMemoryManager).parameters
    kwargs = {"path": ":memory:"} if "path" in params else {}
    mm = mmm.MenaceMemoryManager(**kwargs)
    assert mm._embed("hello") == [0.0]
    assert called["text"] == "hello"


def test_init_local_knowledge_uses_get_embedder(monkeypatch, tmp_path):
    called = {}

    def fake_get():
        called["hit"] = True
        class Dummy:
            def encode(self, arr):
                return [[0.0]]
        return Dummy()

    monkeypatch.setattr(lkm, "get_embedder", fake_get)
    lkm._LOCAL_KNOWLEDGE = None
    lkm.init_local_knowledge(tmp_path / "db.sqlite")
    assert called.get("hit")


def test_get_embedder_initialises_without_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)

    class DummySentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"


def test_get_embedder_exports_token_when_available(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "secret-token")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)

    class DummySentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"
    assert os.environ["HUGGINGFACEHUB_API_TOKEN"] == "secret-token"
    assert os.environ["HF_HUB_TOKEN"] == "secret-token"


def test_initialise_embedder_wait_capped(monkeypatch, caplog):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(
        governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock()
    )
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_TIMEOUT", 9999.0)
    monkeypatch.setattr(governed_embeddings, "_MAX_EMBEDDER_WAIT", 0.5)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_WAIT_CAPPED", False)

    recorded = {}

    class DummyEvent:
        def wait(self, timeout: float) -> bool:
            recorded["timeout"] = timeout
            return False

    monkeypatch.setattr(
        governed_embeddings,
        "_ensure_embedder_thread_locked",
        lambda: DummyEvent(),
    )

    with caplog.at_level(logging.WARNING):
        governed_embeddings._initialise_embedder_with_timeout()

    assert recorded["timeout"] == 0.5
    assert any("capping embedder" in rec.msg for rec in caplog.records)
