import governed_embeddings
from governed_embeddings import governed_embed
import menace.menace_memory_manager as mmm
import menace.local_knowledge_module as lkm
import types


def test_governed_embed_blocks_gpl():
    text = "This code is licensed under the GNU General Public License"
    assert governed_embed(text) is None


def test_governed_embed_redacts_and_embeds(monkeypatch):
    recorded = {}

    def fake_redact(text: str) -> str:
        recorded["redacted"] = text
        return "clean"

    def fake_detect(text: str):
        recorded["detected"] = text
        return None

    class DummyEmbedder:
        def encode(self, arr):
            assert arr[0] == "clean"
            return [types.SimpleNamespace(tolist=lambda: [0.1, 0.2])]

    monkeypatch.setattr(governed_embeddings, "redact_text", fake_redact)
    monkeypatch.setattr(governed_embeddings, "detect", fake_detect)
    monkeypatch.setattr(governed_embeddings, "get_embedder", lambda: DummyEmbedder())

    vec = governed_embed("secret token")
    assert vec == [0.1, 0.2]
    assert recorded["redacted"] == "secret token"
    assert recorded["detected"] == "secret token"


def test_memory_manager_uses_governed_embed(monkeypatch):
    called = {}

    def fake_embed(text: str, embedder=None):
        called["text"] = text
        return [0.0]

    monkeypatch.setattr(mmm, "governed_embed", fake_embed)
    mm = mmm.MenaceMemoryManager(path=":memory:")
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
