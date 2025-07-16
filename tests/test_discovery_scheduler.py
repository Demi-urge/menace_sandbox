import json
import types
from pathlib import Path
import pytest

import menace.discovery_scheduler as ds


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
    def __init__(self) -> None:
        self.called = False

    async def create_bots(self, tasks):
        self.called = True


def fake_run_cycle():
    Path("niche_candidates.json").write_text(
        json.dumps(
            [
                {
                    "platform": "x",
                    "niche": "n",
                    "product_name": "p",
                    "price_point": 1,
                    "tags": ["t"],
                    "trend_signal": 1,
                    "source_url": "u",
                }
            ]
        )
    )


def _stop_after_first(sched):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_scheduler_cycle(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scraper = DummyScraper()
    creator = DummyCreator()
    sched = ds.DiscoveryScheduler(scraper=scraper, creation_bot=creator, interval=0)
    monkeypatch.setattr(ds, "discovery_run_cycle", fake_run_cycle)
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


def test_scheduler_logs_errors(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    scraper = DummyScraper()
    creator = DummyCreator()
    sched = ds.DiscoveryScheduler(scraper=scraper, creation_bot=creator, interval=0)
    monkeypatch.setattr(ds, "discovery_run_cycle", lambda: (_ for _ in ()).throw(RuntimeError("fail2")))
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
