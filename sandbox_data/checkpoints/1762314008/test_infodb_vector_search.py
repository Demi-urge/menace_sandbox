import time
import types
import sys
from types import MethodType

# provide minimal stubs for heavy optional dependencies so InfoDB can be imported
mods = {
    "menace.chatgpt_enhancement_bot": ["EnhancementDB", "ChatGPTEnhancementBot", "Enhancement"],
    "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
    "menace.text_research_bot": ["TextResearchBot"],
    "menace.video_research_bot": ["VideoResearchBot"],
    "menace.chatgpt_research_bot": ["ChatGPTResearchBot", "Exchange", "summarise_text"],
    "menace.database_manager": ["get_connection", "DB_PATH"],
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

import menace.research_aggregator_bot as rab
import pytest


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = rab.InfoDB(
        tmp_path / "i.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"i.{backend}.index",
    )

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "first" in text:
            return [1.0, 0.0]
        if "second" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    item = rab.ResearchItem(
        topic="Topic",
        content="first",
        tags=["t1"],
        associated_bots=["bot"],
        timestamp=time.time(),
    )
    db.add(item)
    assert str(item.item_id) in db._metadata
    assert captured and "tags=t1" in captured[0] and "associated_bots=bot" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0].topic == "Topic"

    # insert record without embedding and backfill
    db.conn.execute(
        "INSERT INTO info(title, tags, content, timestamp) VALUES (?,?,?,?)",
        ("Second", "t2", "second", time.time()),
    )
    db.conn.commit()
    new_id = db.conn.execute("SELECT id FROM info WHERE title='Second'").fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0].title == "Second"
