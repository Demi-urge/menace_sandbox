import time
import types
import sys
import pytest

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


def test_fts_search(tmp_path):
    db = rab.InfoDB(tmp_path / "i.db")
    if not getattr(db, "has_fts", False):
        pytest.skip("fts5 not available")
    item1 = rab.ResearchItem(topic="A", content="alpha", timestamp=time.time())
    db.add(item1)
    item2 = rab.ResearchItem(topic="B", content="bravo", tags=["tag"], timestamp=time.time())
    db.add(item2)
    res1 = db.search("alpha")
    titles = [r.title for r in res1]
    assert "A" in titles
    res2 = db.search("tag")
    titles = [r.title for r in res2]
    assert "B" in titles

