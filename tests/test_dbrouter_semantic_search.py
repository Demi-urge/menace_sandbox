import types
import sys
import time
from types import MethodType

import pytest

# Stub modules for optional heavy dependencies required by InfoDB
mods = {
    "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
    "menace.text_research_bot": ["TextResearchBot"],
    "menace.video_research_bot": ["VideoResearchBot"],
    "menace.chatgpt_research_bot": ["ChatGPTResearchBot", "Exchange", "summarise_text"],
    "menace.database_manager": [
        "get_connection",
        "DB_PATH",
        "process_idea",
        "update_profitability_threshold",
    ],
    "menace.capital_management_bot": ["CapitalManagementBot"],
}
for name, attrs in mods.items():
    module = types.ModuleType(name)
    for attr in attrs:
        if attr == "summarise_text":
            setattr(module, attr, lambda text, ratio=0.2: text[:10])
        elif attr == "get_connection":
            setattr(module, attr, lambda path: None)
        elif attr == "DB_PATH":
            setattr(module, attr, ":memory:")
        else:
            setattr(module, attr, type(attr, (), {}))
    sys.modules.setdefault(name, module)

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB, TelemetryEvent
import menace.chatgpt_enhancement_bot as ceb
import menace.research_aggregator_bot as rab
from menace.database_router import DatabaseRouter
from menace.embeddable_db_mixin import EmbeddableDBMixin


class DummyEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


@pytest.fixture
def shared_embedder(monkeypatch) -> DummyEmbedder:
    embedder = DummyEmbedder()
    monkeypatch.setattr(
        EmbeddableDBMixin, "encode_text", lambda self, text: embedder(text)
    )
    return embedder


def bind_embed(db, embedder: DummyEmbedder) -> None:
    db._embed = MethodType(lambda self, text: embedder(text), db)


def test_semantic_search_router(tmp_path, shared_embedder):
    bot_db = BotDB(path=tmp_path / "bot.db", vector_backend="annoy", vector_index_path=tmp_path / "bot.index")
    workflow_db = WorkflowDB(path=tmp_path / "wf.db", vector_backend="annoy", vector_index_path=tmp_path / "wf.index")
    error_db = ErrorDB(path=tmp_path / "err.db", vector_backend="annoy", vector_index_path=tmp_path / "err.index")
    enhancement_db = ceb.EnhancementDB(tmp_path / "enh.db", vector_index_path=tmp_path / "enh.index")
    info_db = rab.InfoDB(tmp_path / "info.db", vector_backend="annoy", vector_index_path=tmp_path / "info.index")

    for db in [bot_db, workflow_db, error_db, enhancement_db, info_db]:
        bind_embed(db, shared_embedder)

    bot_db.add_bot(BotRecord(name="BotA", purpose="alpha", tags=["x"], toolchain=["tc1"]))
    workflow_db.add(WorkflowRecord(workflow=["alpha"], task_sequence=["s1"], title="Alpha"))
    error_db.add_telemetry(TelemetryEvent(root_cause="alpha", stack_trace="t1"))
    enhancement_db.add(ceb.Enhancement(idea="i1", rationale="r1", summary="alpha", before_code="a", after_code="b"))
    info_db.add(rab.ResearchItem(topic="T1", content="alpha", tags=["t1"], associated_bots=["b1"], timestamp=time.time()))

    router = DatabaseRouter(
        bot_db=bot_db,
        workflow_db=workflow_db,
        info_db=info_db,
        error_db=error_db,
        enhancement_db=enhancement_db,
    )

    results = router.semantic_search("alpha")
    kinds = {r["kind"] for r in results}
    assert kinds == {"bot", "workflow", "error", "enhancement", "info"}
    assert all(r["source_id"] is not None for r in results)
