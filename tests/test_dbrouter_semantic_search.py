import types
import sys
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

from menace.bot_database import BotDB
from menace.task_handoff_bot import WorkflowDB
from menace.error_bot import ErrorDB
import menace.chatgpt_enhancement_bot as ceb
import menace.research_aggregator_bot as rab
from menace.database_router import DatabaseRouter
from menace.universal_retriever import ResultBundle


def test_semantic_search_delegates_to_retriever(tmp_path):
    bot_db = BotDB(path=tmp_path / "bot.db", vector_backend="annoy", vector_index_path=tmp_path / "b.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_backend="annoy", vector_index_path=tmp_path / "w.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_backend="annoy", vector_index_path=tmp_path / "e.idx")
    enh_db = ceb.EnhancementDB(tmp_path / "enh.db", vector_index_path=tmp_path / "enh.idx")
    info_db = rab.InfoDB(tmp_path / "info.db", vector_backend="annoy", vector_index_path=tmp_path / "i.idx")

    router = DatabaseRouter(
        bot_db=bot_db,
        workflow_db=wf_db,
        info_db=info_db,
        error_db=err_db,
        enhancement_db=enh_db,
    )

    called: dict[str, tuple] = {}

    def fake_retrieve(query, top_k=10):
        called["args"] = (query, top_k)
        return [
            ResultBundle(origin_db="bot", metadata={"id": 1}, score=0.5, reason="a"),
            ResultBundle(origin_db="information", metadata={"id": 2}, score=0.4, reason="b"),
        ]

    router._retriever.retrieve = fake_retrieve  # type: ignore

    results = router.semantic_search("alpha", top_k=5)

    assert called["args"] == ("alpha", 5)
    assert [r.origin_db for r in results] == ["bot", "info"]
    assert all(isinstance(r, ResultBundle) for r in results)
    assert results[0].metadata["id"] == 1
