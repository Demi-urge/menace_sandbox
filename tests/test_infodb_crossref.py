import time
import types
import sys

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


def test_cross_reference_notes(tmp_path):
    db = rab.InfoDB(tmp_path / "info.db")
    db.set_current_model(1)
    it1 = rab.ResearchItem(topic="Topic", content="a", timestamp=time.time(), model_id=1)
    db.add(it1)
    it2 = rab.ResearchItem(topic="Topic", content="b", timestamp=time.time(), model_id=1, energy=2)
    db.add(it2)
    it3 = rab.ResearchItem(topic="Topic", content="c", timestamp=time.time(), model_id=2, energy=3)
    db.add(it3)

    items = db.search("Topic")
    by_id = {i.item_id: i for i in items}
    assert "topic" in by_id[it2.item_id].notes
    assert by_id[it2.item_id].corroboration_count == 1
    assert by_id[it3.item_id].corroboration_count == 2
    assert "xref" in by_id[it3.item_id].notes


def test_high_energy_validation(monkeypatch, tmp_path):
    class T:
        def __init__(self):
            self.called = False

        def process(self, urls, files, ratio=0.2):
            self.called = True
            return []

    class V:
        def __init__(self):
            self.called = False

        def process(self, q, ratio=0.2):
            self.called = True
            return []

    text_bot = T()
    video_bot = V()
    db = rab.InfoDB(tmp_path / "info.db", text_bot=text_bot, video_bot=video_bot)
    item = rab.ResearchItem(topic="Topic", content="x", timestamp=time.time(), model_id=1, energy=3)
    db.add(item)
    stored = db.search("Topic")[0]
    assert text_bot.called
    assert video_bot.called
    assert "validated:text" in stored.notes
    assert "validated:video" in stored.notes
    assert stored.corroboration_count == 2
