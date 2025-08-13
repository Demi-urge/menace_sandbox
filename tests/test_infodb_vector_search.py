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


def test_infodb_vector_search(tmp_path):
    db = rab.InfoDB(
        tmp_path / "i.db",
        vector_backend="annoy",
        vector_index_path=tmp_path / "i.index",
    )

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "first" in text:
            return [1.0, 0.0]
        if "updated" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    db._embed = MethodType(fake_embed, db)

    item = rab.ResearchItem(
        topic="Topic",
        content="first",
        tags=["t1"],
        associated_bots=["bot"],
        timestamp=time.time(),
    )
    db.add(item)
    assert captured and "tags:t1" in captured[0] and "associated_bots:bot" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0].topic == "Topic"

    db.update(item.item_id, content="updated")
    assert len(captured) == 2 and "updated" in captured[1]

    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0].content == "updated"
