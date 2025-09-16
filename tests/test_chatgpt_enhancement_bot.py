import json
import logging
import pytest
import sys
import types
import menace
from prompt_types import Prompt
from llm_interface import LLMClient, LLMResult


class DummyBuilder:
    def __init__(self) -> None:
        self.last_prompt: Prompt | None = None

    def refresh_db_weights(self):
        pass

    def build(self, query, **_):
        return ""

    def build_prompt(self, query, *, intent=None, **_):
        meta = {"intent": intent or {}, "vector_confidences": [1.0], "vectors": []}
        self.last_prompt = Prompt(query, metadata=meta, origin="context_builder")
        return self.last_prompt


class RecordingLLM(LLMClient):
    def __init__(self, text: str) -> None:
        super().__init__(model="dummy", log_prompts=False)
        self._text = text
        self.calls = 0
        self.last_prompt: Prompt | None = None
        self.last_builder = None

    def _generate(self, prompt, *, context_builder):  # type: ignore[override]
        self.calls += 1
        self.last_prompt = prompt
        self.last_builder = context_builder
        return LLMResult(text=self._text)


_ctx_mod = types.SimpleNamespace(
    ContextBuilder=DummyBuilder, ErrorResult=Exception, FallbackResult=Exception
)
vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.context_builder = _ctx_mod
embedding_stub = types.SimpleNamespace(
    EmbeddingBackfill=types.SimpleNamespace(watch_events=lambda *a, **k: None)
)
class EmbeddableDBMixin:
    def __init__(self, *a, **k):
        pass

    def backfill_embeddings(self):  # pragma: no cover - simple stub
        pass

    def search_by_vector(self, *a, **k):  # pragma: no cover - simple stub
        return []

vector_service_pkg.embedding_backfill = embedding_stub
vector_service_pkg.EmbeddingBackfill = embedding_stub.EmbeddingBackfill
vector_service_pkg.EmbeddableDBMixin = EmbeddableDBMixin
vector_service_pkg.SharedVectorService = object
vector_service_pkg.CognitionLayer = object
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = _ctx_mod
sys.modules["vector_service.embedding_backfill"] = embedding_stub
menace.RAISE_ERRORS = False
sys.modules["menace.self_coding_engine"] = types.SimpleNamespace(
    MANAGER_CONTEXT=types.SimpleNamespace(set=lambda *a, **k: None, reset=lambda *a, **k: None)
)
sys.modules["menace.self_coding_manager"] = types.SimpleNamespace(
    SelfCodingManager=object
)
sys.modules.setdefault("menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object))
sys.modules.setdefault("shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None))
registry_stub = types.SimpleNamespace(register_bot=lambda *a, **k: None, update_bot=lambda *a, **k: None)
sys.modules.setdefault("menace.bot_registry", types.SimpleNamespace(BotRegistry=lambda: registry_stub))
sys.modules.setdefault("bot_registry", types.SimpleNamespace(BotRegistry=lambda: registry_stub))
decorator_stub = lambda *a, **k: (lambda cls: cls)
sys.modules.setdefault("menace.coding_bot_interface", types.SimpleNamespace(self_coding_managed=decorator_stub))
sys.modules.setdefault("coding_bot_interface", types.SimpleNamespace(self_coding_managed=decorator_stub))

import menace.chatgpt_enhancement_bot as ceb


def test_summarise_text():
    text = "A. B. C."
    summary = ceb.summarise_text(text, ratio=0.34)
    assert "A" in summary
    assert summary.count(".") <= 2


def test_propose(monkeypatch, tmp_path):
    response_text = json.dumps([{"idea": "New", "rationale": "More efficient"}])
    builder = DummyBuilder()

    client = RecordingLLM(response_text)
    client.context_builder = builder  # satisfy ChatGPTEnhancementBot requirements

    router = ceb.init_db_router("enhprop", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(ceb, "SHARED_TABLES", [])
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    db.add_embedding = lambda *a, **k: None
    bot = ceb.ChatGPTEnhancementBot(client, db=db, context_builder=builder)
    monkeypatch.setattr(bot, "_feasible", lambda e: True)
    results = bot.propose(
        "Improve", num_ideas=1, context="ctx", context_builder=builder
    )
    assert results and results[0].context == "ctx"
    assert client.calls == 1
    assert client.last_prompt is builder.last_prompt
    assert builder.last_prompt is not None
    assert builder.last_prompt.origin == "context_builder"
    assert client.last_builder is builder


def test_propose_requires_explicit_context_builder(monkeypatch, tmp_path):
    response_text = json.dumps([{"idea": "New", "rationale": "More efficient"}])
    builder = DummyBuilder()
    client = RecordingLLM(response_text)
    client.context_builder = builder

    router = ceb.init_db_router(
        "enhprop_missing",
        str(tmp_path / "local.db"),
        str(tmp_path / "shared.db"),
    )
    monkeypatch.setattr(ceb, "SHARED_TABLES", [])
    db = ceb.EnhancementDB(tmp_path / "enh_missing.db", router=router)
    db.add_embedding = lambda *a, **k: None
    bot = ceb.ChatGPTEnhancementBot(client, db=db, context_builder=builder)

    with pytest.raises(ValueError, match="context_builder is required"):
        bot.propose("Improve", num_ideas=1, context_builder=None)


def test_propose_requires_context_builder_origin(monkeypatch, tmp_path):
    response_text = json.dumps([{"idea": "New", "rationale": "More efficient"}])

    class BadBuilder(DummyBuilder):
        def build_prompt(self, query, *, intent=None, **_):
            meta = {"intent": intent or {}, "vector_confidences": [1.0], "vectors": []}
            self.last_prompt = Prompt(query, metadata=meta)
            return self.last_prompt

    builder = BadBuilder()
    client = RecordingLLM(response_text)
    client.context_builder = builder

    router = ceb.init_db_router("enhprop_bad", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(ceb, "SHARED_TABLES", [])
    db = ceb.EnhancementDB(tmp_path / "enh_bad.db", router=router)
    db.add_embedding = lambda *a, **k: None
    bot = ceb.ChatGPTEnhancementBot(client, db=db, context_builder=builder)
    results = bot.propose("Improve", num_ideas=1, context_builder=builder)
    assert results == []
    assert builder.last_prompt is not None
    assert builder.last_prompt.origin == ""
    assert client.calls == 0


def test_requires_client_context_builder(tmp_path):
    class DummyClient:
        context_builder = None

    builder = DummyBuilder()
    db = ceb.EnhancementDB(tmp_path / "e.db")
    with pytest.raises(ValueError):
        ceb.ChatGPTEnhancementBot(DummyClient(), db=db, context_builder=builder)


def test_enhancementdb_duplicate(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(ceb, "SHARED_TABLES", [])
    db = ceb.EnhancementDB(tmp_path / "enh.db")
    db.add_embedding = lambda *a, **k: None

    captured: dict[str, int | None] = {"id": None}
    orig = ceb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(ceb, "insert_if_unique", wrapper)

    enh = ceb.Enhancement(idea="i", rationale="r")
    first = db.add(enh)
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add(ceb.Enhancement(idea="i", rationale="r"))
    assert first == second
    assert captured["id"] == first


def test_enhancementdb_content_hash_unique_index(tmp_path):
    old = ceb.GLOBAL_ROUTER
    router = ceb.init_db_router("enhidx", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    with db._connect() as conn:
        indexes = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA index_list('enhancements')").fetchall()
        }
    assert indexes.get("idx_enhancements_content_hash") == 1
    ceb.GLOBAL_ROUTER = old

