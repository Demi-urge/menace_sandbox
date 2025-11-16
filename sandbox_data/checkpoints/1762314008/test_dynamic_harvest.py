import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64
import pytest
from neurosales.dynamic_harvest import DynamicHarvester
from unittest.mock import patch


class FakeElement:
    def __init__(self, text):
        self.text = text

    async def evaluate(self, script):
        return self.text

    async def inner_text(self):
        return self.text


class FakePage:
    def __init__(self):
        self.scrolls = 0

    async def goto(self, url):
        pass

    async def fill(self, sel, val):
        pass

    async def click(self, sel):
        pass

    async def wait_for_selector(self, sel):
        pass

    async def query_selector_all(self, sel):
        if sel == "svg":
            return [FakeElement("<svg>A</svg>"), FakeElement("<svg>B</svg>")]
        if sel == "article":
            if self.scrolls == 0:
                return [FakeElement("p1"), FakeElement("p2")]
            return [FakeElement("p1"), FakeElement("p2"), FakeElement("p3")]
        return []

    async def evaluate(self, script):
        self.scrolls += 1
        return self.scrolls * 1000

    async def wait_for_load_state(self, state="networkidle"):
        pass


class FakeContext:
    async def new_page(self):
        return FakePage()


class FakeBrowser:
    async def new_context(self, **kwargs):
        return FakeContext()

    async def close(self):
        pass


class FakeChromium:
    async def launch(self, headless=True):
        return FakeBrowser()


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


def fake_async_playwright():
    return FakePlaywright()


def test_harvest_dashboard():
    harvester = DynamicHarvester()
    with patch("neurosales.dynamic_harvest.async_playwright", fake_async_playwright), \
        patch("neurosales.dynamic_harvest.ensure_playwright_browsers", lambda: True):
        data = harvester.harvest_dashboard("http://x", "u", "p")
    assert len(data) == 2
    assert base64.b64decode(data[0]).startswith(b"<svg")


def test_harvest_infinite_scroll():
    harvester = DynamicHarvester()
    with patch("neurosales.dynamic_harvest.async_playwright", fake_async_playwright), \
        patch("neurosales.dynamic_harvest.ensure_playwright_browsers", lambda: True):
        items = harvester.harvest_infinite_scroll("http://x")
    assert set(items) == {"p1", "p2", "p3"}


def test_env_proxy_list_parsed(monkeypatch):
    monkeypatch.setenv("NEURO_PROXY_LIST", "http://a,http://b")
    harvester = DynamicHarvester()
    assert set(harvester.proxies) == {"http://a", "http://b"}


def test_harvest_dashboard_warns_when_unavailable(caplog):
    harvester = DynamicHarvester()
    caplog.set_level("WARNING")
    with patch("neurosales.dynamic_harvest.async_playwright", None):
        data = harvester.harvest_dashboard("http://x", "u", "p")
    assert data == []
    assert any("playwright unavailable" in r.getMessage() for r in caplog.records)


def test_harvest_infinite_scroll_warns_when_unavailable(caplog):
    harvester = DynamicHarvester()
    caplog.set_level("WARNING")
    with patch("neurosales.dynamic_harvest.async_playwright", None):
        items = harvester.harvest_infinite_scroll("http://x")
    assert items == []
    assert any("playwright unavailable" in r.getMessage() for r in caplog.records)

