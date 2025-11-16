import json
import types
from pathlib import Path
import pytest
import sys

# Stub heavy modules to keep tests lightweight
mde = types.ModuleType("menace.menace_discovery_engine")
async def _stub_run_cycle(*, context_builder):
    pass
mde.run_cycle = _stub_run_cycle
sys.modules.setdefault("menace.menace_discovery_engine", mde)

rab = types.ModuleType("menace.research_aggregator_bot")
class _InfoDB:
    def add(self, *a, **k):
        pass
class _ResearchItem:
    def __init__(self, *a, **k):
        pass
rab.InfoDB = _InfoDB
rab.ResearchItem = _ResearchItem
sys.modules.setdefault("menace.research_aggregator_bot", rab)

bcb = types.ModuleType("menace.bot_creation_bot")
class _BotCreationBot:
    def __init__(self, *, context_builder):
        self.context_builder = context_builder
        self.called = False

    async def create_bots(self, tasks):
        self.called = True
bcb.BotCreationBot = _BotCreationBot
sys.modules.setdefault("menace.bot_creation_bot", bcb)

bpb = types.ModuleType("menace.bot_planning_bot")
class _PlanningTask:
    def __init__(self, *a, **k):
        pass
bpb.PlanningTask = _PlanningTask
sys.modules.setdefault("menace.bot_planning_bot", bpb)

nsd = types.ModuleType("menace.normalize_scraped_data")
nsd.load_items = lambda paths: []
nsd.normalize = lambda data: []
sys.modules.setdefault("menace.normalize_scraped_data", nsd)

ts = types.ModuleType("menace.trending_scraper")
class _TrendingScraper:
    def scrape_reddit(self, energy=None):
        return []
class _TrendingItem:
    pass
ts.TrendingScraper = _TrendingScraper
ts.TrendingItem = _TrendingItem
sys.modules.setdefault("menace.trending_scraper", ts)

vs = types.ModuleType("vector_service")
class _ContextBuilder:
    def refresh_db_weights(self):
        pass
vs.ContextBuilder = _ContextBuilder
sys.modules.setdefault("vector_service", vs)
vs_cb = types.ModuleType("vector_service.context_builder")
vs_cb.ContextBuilder = _ContextBuilder
sys.modules.setdefault("vector_service.context_builder", vs_cb)

import menace.discovery_scheduler as ds


class DummyBuilder:
    def __init__(self) -> None:
        self.refreshed = False

    def refresh_db_weights(self) -> None:
        self.refreshed = True


class DummyScraper:
    def __init__(self) -> None:
        self.called = False

    def scrape_reddit(self, energy=None):
        self.called = True
        return [
            types.SimpleNamespace(
                platform="x",
                niche="n",
                product_name="p",
                price_point=None,
                tags=["t"],
                trend_signal=1.0,
                source_url="u",
            )
        ]


class DummyCreator:
    def __init__(self, context_builder):
        self.called = False
        self.context_builder = context_builder

    async def create_bots(self, tasks):
        self.called = True


async def fake_run_cycle(*, context_builder):
    pass


def _stop_after_first(sched):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_scheduler_cycle(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scraper = DummyScraper()
    builder = DummyBuilder()
    creator = DummyCreator(builder)
    sched = ds.DiscoveryScheduler(
        scraper=scraper, creation_bot=creator, context_builder=builder, interval=0
    )
    monkeypatch.setattr(ds, "discovery_run_cycle", fake_run_cycle)
    monkeypatch.setattr(ds.DiscoveryScheduler, "_new_candidates", lambda self: ["x"])
    monkeypatch.setattr(ds.DiscoveryScheduler, "_save_items", lambda self, db, items: None)
    monkeypatch.setattr(ds.time, "sleep", _stop_after_first(sched))

    class DummyDB:
        def __init__(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

    monkeypatch.setattr(ds, "InfoDB", DummyDB)
    sched.running = True
    with pytest.raises(SystemExit):
        sched._loop()

    assert scraper.called
    assert creator.called
    assert builder.refreshed
    assert sched.context_builder is builder


def test_scheduler_logs_errors(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    scraper = DummyScraper()
    builder = DummyBuilder()
    creator = DummyCreator(builder)
    sched = ds.DiscoveryScheduler(
        scraper=scraper, creation_bot=creator, context_builder=builder, interval=0
    )
    monkeypatch.setattr(
        ds,
        "discovery_run_cycle",
        lambda **k: (_ for _ in ()).throw(RuntimeError("fail2")),
    )
    monkeypatch.setattr(scraper, "scrape_reddit", lambda energy=None: (_ for _ in ()).throw(RuntimeError("fail1")))
    async def boom(tasks):
        raise RuntimeError("fail3")
    monkeypatch.setattr(creator, "create_bots", boom)
    monkeypatch.setattr(ds.DiscoveryScheduler, "_new_candidates", lambda self: ["x"])
    monkeypatch.setattr(ds.time, "sleep", _stop_after_first(sched))

    class DummyDB:
        def add(self, *a, **k):
            pass

    monkeypatch.setattr(ds, "InfoDB", lambda: DummyDB())
    sched.running = True
    caplog.set_level("ERROR")
    with pytest.raises(SystemExit):
        sched._loop()

    text = caplog.text
    assert "scraper failed" in text
    assert "discovery run failed" in text
    assert "bot creation failed" in text
    assert builder.refreshed
    assert sched.context_builder is builder


def test_scheduler_injects_builder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    builder = DummyBuilder()

    class DummyCreatorBot:
        def __init__(self, *, context_builder):
            self.context_builder = context_builder
            self.called = False

        async def create_bots(self, tasks):
            self.called = True

    monkeypatch.setattr(ds, "BotCreationBot", DummyCreatorBot)
    sched = ds.DiscoveryScheduler(
        scraper=DummyScraper(), context_builder=builder, interval=0
    )

    assert isinstance(sched.creation_bot, DummyCreatorBot)
    assert sched.creation_bot.context_builder is builder
    assert builder.refreshed
