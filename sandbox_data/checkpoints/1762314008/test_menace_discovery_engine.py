import asyncio
import os
import sys
from types import SimpleNamespace, ModuleType

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

stub_news = ModuleType("menace.newsreader_bot")
from dataclasses import dataclass

@dataclass
class Event:
    title: str
    source: str
    categories: list[str]
    impact: float
    summary: str = ""

def fetch_news(*a, **k):
    return []

stub_news.Event = Event
stub_news.fetch_news = fetch_news
sys.modules["menace.newsreader_bot"] = stub_news

stub_cap = ModuleType("menace.capital_management_bot")
class CapBot:
    def energy_score(self, *a, **k):
        return 0.5
stub_cap.CapitalManagementBot = CapBot
sys.modules["menace.capital_management_bot"] = stub_cap

stub_passive = ModuleType("menace.passive_discovery_bot")
class PassiveDiscoveryBot:
    async def collect(self, *a, **k):
        return []
stub_passive.PassiveDiscoveryBot = PassiveDiscoveryBot
sys.modules["menace.passive_discovery_bot"] = stub_passive

stub_dbm = ModuleType("menace.database_management_bot")
class DBBot:
    def ingest_idea(self, *a, **k):
        pass
    def adjust_threshold(self):
        pass
stub_dbm.DatabaseManagementBot = DBBot
sys.modules["menace.database_management_bot"] = stub_dbm

stub_sat = ModuleType("menace.niche_saturation_bot")
class NCand:
    def __init__(self, name="", demand=0.0, competition=0.0, trend=0.0):
        self.product_name = name
        self.niche = ""
        self.platform = ""
        self.tags = []
        self.trend_signal = trend

class SatBot:
    def __init__(self, *a, **k):
        pass

    def detect(self, items):
        return items

stub_sat.NicheCandidate = NCand
stub_sat.NicheSaturationBot = SatBot
sys.modules["menace.niche_saturation_bot"] = stub_sat

stub_ctx = ModuleType("vector_service.context_builder")
class ContextBuilder:
    def __init__(self, *a, **k):
        pass
stub_ctx.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = stub_ctx

stub_match = ModuleType("menace.candidate_matcher")
stub_match.find_matching_models = lambda c: False
sys.modules["menace.candidate_matcher"] = stub_match

stub_norm = ModuleType("menace.normalize_scraped_data")
class NormCand:
    def __init__(self, *a, **k):
        pass
stub_norm.load_items = lambda paths: []
stub_norm.normalize = lambda data: []
stub_norm.save_candidates = lambda p, c: None
stub_norm.NicheCandidate = NormCand
sys.modules["menace.normalize_scraped_data"] = stub_norm

stub_research = ModuleType("menace.research_aggregator_bot")
class ResearchItem:
    def __init__(self, *a, **k):
        pass
class InfoDB:
    def add(self, item):
        pass
stub_research.InfoDB = InfoDB
stub_research.ResearchItem = ResearchItem
sys.modules["menace.research_aggregator_bot"] = stub_research


import menace.menace_discovery_engine as mde

class DummyDB:
    def add(self, item):
        pass

class DummyBot:
    def __init__(self):
        self.ingested = []
    def ingest_idea(self, title, *, tags=(), source="", urls=()):
        self.ingested.append(title)

async def _run(db, bot, energy=0.3):
    return await mde.gather_search_ideas(db, bot, energy=energy)


def test_gather_search_ideas_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_ENGINE_ID", "e")
    monkeypatch.setattr(mde.isb, "requests", object())

    created = {}
    class DummyClient:
        def __init__(self, key, engine):
            created["creds"] = (key, engine)
    monkeypatch.setattr(mde.isb, "GoogleSearchClient", DummyClient)
    res = mde.isb.Result(title="ModelA", link="u", snippet="s")
    monkeypatch.setattr(mde.isb, "discover_new_models", lambda c, b, energy=0: [res])
    monkeypatch.setattr(mde, "_store_info", lambda *a, **k: None)
    bot = DummyBot()
    out = asyncio.run(_run(DummyDB(), bot))
    assert out == ["ModelA"]
    assert bot.ingested == ["ModelA"]
    assert created["creds"] == ("k", "e")


def test_gather_search_ideas_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_ENGINE_ID", raising=False)
    monkeypatch.setattr(mde.isb, "requests", None)

    class DummyScraper:
        def collect_all(self, energy=None):
            return [SimpleNamespace(product_name="ModelB", tags=["ai"], platform="p", source_url="http://b")]
    monkeypatch.setattr(mde, "TrendingScraper", lambda: DummyScraper())
    monkeypatch.setattr(mde.isb.KeywordBank, "generate_queries", lambda self, energy: ["modelb"])
    monkeypatch.setattr(mde, "_store_info", lambda *a, **k: None)
    bot = DummyBot()
    out = asyncio.run(_run(DummyDB(), bot))
    assert out == ["ModelB"]
    assert bot.ingested == ["ModelB"]
