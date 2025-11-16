import sqlite3
import sys
import types
from pathlib import Path

# Stub heavy dependencies imported by chatgpt_idea_bot
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

import menace_sandbox.chatgpt_idea_bot as cib
from prompt_types import Prompt


class DBContextBuilder:
    """Minimal builder that aggregates snippets from local SQLite DBs."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.calls: list[str] = []

    def refresh_db_weights(self) -> None:  # pragma: no cover - no-op for tests
        pass

    def build(self, query: str, **_: object) -> str:
        self.calls.append(query)
        parts: list[str] = []
        for p in self.paths:
            with sqlite3.connect(p) as conn:
                row = conn.execute("SELECT content FROM snippets LIMIT 1").fetchone()
                if row:
                    parts.append(row[0])
        return "\n".join(parts)

    def build_prompt(self, query, *, intent_metadata=None, prior=None, **_):
        session_id = "sid"
        tags = list((intent_metadata or {}).get("tags", []) or [])
        if isinstance(query, (list, tuple)):
            query_text = " ".join(str(q) for q in query)
        else:
            query_text = str(query)
        context_raw = self.build(
            " ".join(tags) if tags else query_text,
            session_id=session_id,
        )
        context = context_raw if isinstance(context_raw, str) else ""
        parts = [prior, context, query_text]
        user = "\n".join(p for p in parts if p)
        meta = {"retrieval_session_id": session_id, "origin": "context_builder"}
        if tags:
            meta["tags"] = list(tags)
            meta["intent_tags"] = list(tags)
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


def _init_db(path: Path, content: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE snippets(content TEXT)")
        conn.execute("INSERT INTO snippets VALUES (?)", (content,))


def test_prompts_embed_local_db_data(tmp_path):
    # Prepare lightweight DBs with identifiable content
    db_contents = {
        "bots.db": "bot-info",
        "code.db": "code-snippet",
        "errors.db": "error-log",
        "workflows.db": "workflow-step",
    }
    paths = []
    for name, text in db_contents.items():
        db_path = tmp_path / name
        _init_db(db_path, text)
        paths.append(db_path)

    builder = DBContextBuilder(paths)
    client = cib.ChatGPTClient(context_builder=builder)
    prompt = client.build_prompt_with_memory(["x"], prior="y", context_builder=builder)

    content = prompt.user
    for text in db_contents.values():
        assert text in content

