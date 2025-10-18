import inspect
import logging
import sys
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


def test_get_embedder_skips_login_without_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)

    class DummySentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

    def raising_login(*_args, **_kwargs):
        raise AssertionError("login should not be called when token is missing")

    fake_module = types.SimpleNamespace(login=raising_login)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"


def test_get_embedder_logs_in_when_token_available(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "secret-token")
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)

    recorded: dict[str, object] = {}

    def fake_login(*_args, **kwargs):
        recorded["token"] = kwargs.get("token")

    fake_module = types.SimpleNamespace(login=fake_login)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    class DummySentenceTransformer:
        def __init__(self, name: str) -> None:
            recorded["model"] = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    governed_embeddings.get_embedder()
    assert recorded["token"] == "secret-token"
    assert recorded["model"] == "all-MiniLM-L6-v2"
