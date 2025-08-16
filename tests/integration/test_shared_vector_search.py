import time
import types
import sys
from types import MethodType
import pytest

# provide minimal stubs for heavy optional dependencies so InfoDB can be imported
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
from menace.error_bot import ErrorDB
from menace.error_logger import TelemetryEvent
from menace.universal_retriever import UniversalRetriever
import menace.chatgpt_enhancement_bot as ceb
import menace.research_aggregator_bot as rab


class DummyEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


@pytest.fixture
def shared_embedder() -> DummyEmbedder:
    return DummyEmbedder()


def bind_embed(db, embedder: DummyEmbedder) -> None:
    db.encode_text = MethodType(lambda self, text: embedder(text), db)


def test_search_by_vector_shared_backend(tmp_path, shared_embedder):
    bot_db = BotDB(path=tmp_path / "bot.db", vector_backend="annoy", vector_index_path=tmp_path / "bot.index")
    workflow_db = WorkflowDB(path=tmp_path / "wf.db", vector_backend="annoy", vector_index_path=tmp_path / "wf.index")
    error_db = ErrorDB(path=tmp_path / "err.db", vector_backend="annoy", vector_index_path=tmp_path / "err.index")
    enhancement_db = ceb.EnhancementDB(tmp_path / "enh.db", vector_index_path=tmp_path / "enh.index")
    info_db = rab.InfoDB(tmp_path / "info.db", vector_backend="annoy", vector_index_path=tmp_path / "info.index")

    for db in [bot_db, workflow_db, error_db, enhancement_db, info_db]:
        bind_embed(db, shared_embedder)

    # Bot database
    bot_db.add_bot(BotRecord(name="BotA", purpose="alpha", tags=["x"], toolchain=["tc1"]))
    bot_db.add_bot(BotRecord(name="BotB", purpose="beta", tags=["y"], toolchain=["tc2"]))
    assert bot_db.search_by_vector([1.0, 0.0], top_k=1)[0]["name"] == "BotA"
    assert bot_db.search_by_vector([0.0, 1.0], top_k=1)[0]["name"] == "BotB"

    # Workflow database
    workflow_db.add(WorkflowRecord(workflow=["alpha"], task_sequence=["s1"], title="Alpha"))
    workflow_db.add(WorkflowRecord(workflow=["beta"], task_sequence=["s2"], title="Beta"))
    assert workflow_db.search_by_vector([1.0, 0.0], top_k=1)[0].title == "Alpha"
    assert workflow_db.search_by_vector([0.0, 1.0], top_k=1)[0].title == "Beta"

    # Error database
    error_db.add_telemetry(TelemetryEvent(root_cause="alpha", stack_trace="trace1"))
    error_db.add_telemetry(TelemetryEvent(root_cause="beta", stack_trace="trace2"))
    assert error_db.search_by_vector([1.0, 0.0], top_k=1)[0]["cause"] == "alpha"
    assert error_db.search_by_vector([0.0, 1.0], top_k=1)[0]["cause"] == "beta"

    # Enhancement database
    enhancement_db.add(ceb.Enhancement(idea="i1", rationale="r1", summary="alpha", before_code="a", after_code="b"))
    enhancement_db.add(ceb.Enhancement(idea="i2", rationale="r2", summary="beta", before_code="c", after_code="d"))
    assert enhancement_db.search_by_vector([1.0, 0.0], top_k=1)[0].summary == "alpha"
    assert enhancement_db.search_by_vector([0.0, 1.0], top_k=1)[0].summary == "beta"

    # Info database
    info_db.add(rab.ResearchItem(topic="T1", content="alpha", tags=["t1"], associated_bots=["b1"], timestamp=time.time()))
    info_db.add(rab.ResearchItem(topic="T2", content="beta", tags=["t2"], associated_bots=["b2"], timestamp=time.time()))
    assert info_db.search_by_vector([1.0, 0.0], top_k=1)[0].content == "alpha"
    assert info_db.search_by_vector([0.0, 1.0], top_k=1)[0].content == "beta"

    # Universal retriever pulls from all databases via a shared interface
    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=workflow_db,
        error_db=error_db,
        enhancement_db=enhancement_db,
        information_db=info_db,
    )
    hits, session_id, vectors = retriever.retrieve("alpha", top_k=5)
    assert session_id and vectors
    assert {h.origin_db for h in hits} == {"bot", "workflow", "error", "enhancement", "information"}

    # ensure shared embedder was used across databases
    assert len(shared_embedder.calls) >= 10
