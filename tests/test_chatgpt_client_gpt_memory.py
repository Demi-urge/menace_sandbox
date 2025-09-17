import sys
import types

# stub modules required by ChatGPTClient
sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace_sandbox.database_management_bot",
    types.SimpleNamespace(DatabaseManagementBot=object),
)
sys.modules.setdefault(
    "menace_sandbox.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace_sandbox.memory_logging", types.SimpleNamespace(log_with_tags=lambda *a, **k: None)
)
sys.modules.setdefault(
    "menace_sandbox.memory_aware_gpt_client",
    types.SimpleNamespace(ask_with_memory=lambda *a, **k: {}),
)
sys.modules.setdefault(
    "menace_sandbox.local_knowledge_module",
    types.SimpleNamespace(LocalKnowledgeModule=lambda *a, **k: types.SimpleNamespace(memory=None)),
)
sys.modules.setdefault(
    "menace_sandbox.knowledge_retriever",
    types.SimpleNamespace(
        get_feedback=lambda *a, **k: [],
        get_improvement_paths=lambda *a, **k: [],
        get_error_fixes=lambda *a, **k: [],
    ),
)
sys.modules.setdefault(
    "governed_retrieval",
    types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x),
)
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(SharedVectorService=object),
)

# stub sentence_transformers to avoid heavy import
stub_st = types.ModuleType("sentence_transformers")


class _DummyModel:
    def encode(self, text):
        return [0.0]


stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402
from prompt_types import Prompt  # noqa: E402


class StubMemory:
    def __init__(self):
        self.entries: list[types.SimpleNamespace] = []

    def log_interaction(self, prompt, response, tags):
        self.entries.append(
            types.SimpleNamespace(prompt=prompt, response=response, tags=list(tags))
        )

    def fetch_context(self, tags):
        for e in reversed(self.entries):
            if set(tags) & set(e.tags):
                return e.prompt
        return ""

    def search_context(self, query, tags=None):
        if tags:
            return [e for e in self.entries if set(tags) & set(e.tags)]
        return list(self.entries)

    def compact(self, limits):
        for tag, limit in limits.items():
            tagged = [e for e in self.entries if tag in e.tags]
            if len(tagged) > limit:
                for e in tagged[:-limit]:
                    if "summary" not in e.tags:
                        e.tags.append("summary")


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, query, **_):
        return ""

    def build_prompt(self, query, *, intent_metadata=None, prior=None, **_):
        session_id = "sid"
        tags = list((intent_metadata or {}).get("tags", []) or [])
        if isinstance(query, (list, tuple)):
            query_text = " ".join(str(q) for q in query)
        else:
            query_text = str(query)
        effective_tags = list(tags) if tags else [query_text]
        context_raw = self.build(
            " ".join(tags) if tags else query_text,
            session_id=session_id,
        )
        context = context_raw if isinstance(context_raw, str) else ""
        memory_ctx = ""
        mem = getattr(self, "memory", None)
        if mem:
            fetch = getattr(mem, "fetch_context", None)
            if callable(fetch):
                memory_ctx = fetch(tags)
            else:
                search = getattr(mem, "search_context", None)
                if callable(search):
                    entries = search("", tags=tags)
                    if entries:
                        first = entries[0]
                        memory_ctx = getattr(first, "prompt", "") or getattr(first, "response", "")
        parts = [prior, memory_ctx, context, query_text]
        user = "\n".join(p for p in parts if p)
        meta = {"retrieval_session_id": session_id, "origin": "context_builder"}
        if tags:
            meta["tags"] = list(tags)
        if effective_tags:
            meta["intent_tags"] = list(effective_tags)
        if intent_metadata:
            extra_meta = dict(intent_metadata)
            extra_meta.pop("tags", None)
            meta.update(extra_meta)
        return Prompt(
            user,
            system="",
            examples=[],
            tags=list(tags),
            metadata=meta,
            origin="context_builder",
        )


def test_build_prompt_injects_summary_and_logs(monkeypatch):
    mem = StubMemory()
    mem.log_interaction("early prompt", "early resp", ["topic"])

    builder = DummyBuilder()
    builder.memory = mem
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    client.session = None  # offline mode
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "later resp"}}]},
    )
    monkeypatch.setattr(
        cib,
        "log_with_tags",
        lambda memobj, prompt, response, tags: memobj.log_interaction(
            prompt, response, tags
        ),
    )

    prompt = client.build_prompt_with_memory(
        ["topic"], prior="new question", context_builder=client.context_builder
    )
    assert "early prompt" in prompt.user

    client.generate(
        prompt,
        context_builder=client.context_builder,
        tags=["topic"],
    )
    entries = mem.search_context("new question")
    assert any(e.response == "later resp" for e in entries)


def test_summarize_and_prune_via_client(monkeypatch):
    mem = StubMemory()
    builder = DummyBuilder()
    builder.memory = mem
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    client.session = None
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )
    monkeypatch.setattr(
        cib,
        "log_with_tags",
        lambda memobj, prompt, response, tags: memobj.log_interaction(
            prompt, response, tags
        ),
    )

    for i in range(3):
        client.ask([{"role": "user", "content": f"ask{i}"}], tags=["insight"])

    assert len(mem.search_context("", tags=["insight"])) == 3
    mem.compact({"insight": 1})
    remaining = [
        e for e in mem.search_context("", tags=["insight"]) if "summary" not in e.tags
    ]
    assert len(remaining) == 1 and "ask2" in remaining[0].prompt
    summaries = [
        e for e in mem.search_context("", tags=["insight"]) if "summary" in e.tags
    ]
    assert summaries
