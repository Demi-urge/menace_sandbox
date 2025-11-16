import pytest
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types
sys.modules["menace.db_router"] = types.SimpleNamespace(DBRouter=object)
import menace.text_research_bot as trb


def test_summarise_text(monkeypatch):
    monkeypatch.setattr(trb, "summarize", None)
    text = (
        "Common words repeat repeat. Unique phrase stands alone. Another repeat repeat sentence."
    )
    summary = trb.summarise_text(text, ratio=0.34)
    assert "Unique phrase stands alone" in summary
    assert summary.count(".") <= 2


def test_extract_from_urls(monkeypatch):
    pytest.skip("optional dependencies not installed")
    html = "<html><body><p>Hello world from page with content</p></body></html>"
    monkeypatch.setattr(trb, "_download", lambda session, url: html)
    bot = trb.TextResearchBot()
    items = bot.extract_from_urls(["http://example.com"])
    assert items and "Hello world" in items[0].content
    assert items[0].url == "http://example.com"


def test_process(monkeypatch, tmp_path: Path):
    pytest.skip("optional dependencies not installed")
    html = "<html><body>Many words here for summary extraction. Another sentence.</body></html>"
    monkeypatch.setattr(trb, "_download", lambda session, url: html)
    monkeypatch.setattr(trb, "_parse_pdf", lambda path: "Lots of content in document for testing.")
    sent = []
    monkeypatch.setattr(trb, "send_to_aggregator", lambda items: sent.extend(items))

    file = tmp_path / "a.pdf"
    file.write_bytes(b"%PDF-1.4\n%fake pdf")

    bot = trb.TextResearchBot()
    result = bot.process(["http://example.com"], [file], ratio=0.5)
    assert sent == result
    assert len(result) == 2
    assert all(r.content for r in result)


def test_db_hit_avoids_download(monkeypatch):
    pytest.skip("optional dependencies not installed")
    from types import SimpleNamespace
    import time
    from menace.research_aggregator_bot import ResearchItem
    
    router = SimpleNamespace()
    router.query_all = lambda term: SimpleNamespace(code=[], bots=[], info=[ResearchItem(topic=term, content="cached", timestamp=time.time(), source_url="u")], memory=[], menace=[])
    bot = trb.TextResearchBot(db_router=router)

    def fail(*a, **k):
        raise AssertionError("downloaded")

    monkeypatch.setattr(trb, "_download", fail)
    res = bot.process(["Topic"], [])
    assert res and res[0].content == "cached"

