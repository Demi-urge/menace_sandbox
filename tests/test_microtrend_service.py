from menace.microtrend_service import MicrotrendService
import threading
import time
from menace.trending_scraper import TrendingItem, TrendingScraper
import logging


class DummyScraper:
    def collect_all(self):
        return [
            TrendingItem(platform="A", product_name="Foo"),
            TrendingItem(platform="B", product_name="Foo"),
        ]


class DummyKG:
    def __init__(self):
        self.items = []

    def add_trending_item(self, name: str):
        self.items.append(name)


def test_run_once():
    kg = DummyKG()
    svc = MicrotrendService(scraper=DummyScraper(), knowledge=kg)
    svc.run_once()
    assert "Foo" in kg.items


def test_scheduler_invokes(monkeypatch):
    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG())
    calls = []
    monkeypatch.setattr(svc, "run_once", lambda: calls.append(1))
    stop = threading.Event()
    thread = svc.run_continuous(interval=0.0, stop_event=stop)
    time.sleep(0.01)
    stop.set()
    thread.join(timeout=0.1)
    assert calls


def test_planner_called_for_new(monkeypatch):
    class SeqScraper:
        def __init__(self):
            self.calls = 0

        def collect_all(self):
            self.calls += 1
            if self.calls == 1:
                return [
                    TrendingItem(platform="A", product_name="Foo"),
                    TrendingItem(platform="B", product_name="Foo"),
                ]
            return [
                TrendingItem(platform="A", product_name="Foo"),
                TrendingItem(platform="B", product_name="Foo"),
                TrendingItem(platform="A", product_name="Bar"),
                TrendingItem(platform="B", product_name="Bar"),
            ]

    class DummyPlanner:
        def __init__(self):
            self.calls = 0

        def __call__(self, items):
            self.calls += 1

    monkeypatch.setattr(TrendingScraper, "detect_microtrends", lambda items: items)
    planner = DummyPlanner()
    svc = MicrotrendService(scraper=SeqScraper(), knowledge=DummyKG(), planner=planner)
    svc.run_once()
    svc.run_once()
    assert planner.calls == 2


def test_hint_published(monkeypatch):
    monkeypatch.setattr(TrendingScraper, "detect_microtrends", lambda items: items)

    class DummyOrch:
        def __init__(self):
            self.hints = []

        def receive_scaling_hint(self, hint):
            self.hints.append(hint)

    orch = DummyOrch()
    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG(), orchestrator=orch)
    svc.run_once()
    assert orch.hints == ["scale_up"]


def test_on_new_error_logged(monkeypatch, caplog):
    def boom(items):
        raise RuntimeError("fail")

    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG(), on_new=boom)
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert "on_new callback failed" in caplog.text


def test_planner_error_logged(monkeypatch, caplog):
    class BadPlanner:
        def __call__(self, items):
            raise RuntimeError("boom")

    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG(), planner=BadPlanner())
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert "planner failed" in caplog.text


def test_event_bus_retry(monkeypatch, caplog):
    monkeypatch.setattr(TrendingScraper, "detect_microtrends", lambda items: items)

    class DummyBus:
        def __init__(self):
            self.calls = 0

        def publish(self, topic, event):
            self.calls += 1
            raise RuntimeError("boom")

    bus = DummyBus()
    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG(), event_bus=bus)
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert bus.calls == 3
    assert "failed to send scaling hint" in caplog.text


def test_orchestrator_retry(monkeypatch, caplog):
    monkeypatch.setattr(TrendingScraper, "detect_microtrends", lambda items: items)

    class BadOrch:
        def __init__(self):
            self.calls = 0

        def receive_scaling_hint(self, hint):
            self.calls += 1
            raise RuntimeError("nope")

    orch = BadOrch()
    svc = MicrotrendService(scraper=DummyScraper(), knowledge=DummyKG(), orchestrator=orch)
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert orch.calls == 3
    assert "failed to send scaling hint" in caplog.text
