import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.competitive_intelligence_bot as cib


def test_sentiment_basic():
    pos = cib.analyse_sentiment("Great profit ahead")
    neg = cib.analyse_sentiment("Bad loss ahead")
    assert pos > neg


def test_extract_entities():
    ents = cib.extract_entities("Apple releases new iPhone in California")
    assert "Apple" in ents


def test_detect_ai_signals():
    up = cib.CompetitorUpdate(
        title="AI powered update",
        content="company adds new GPT features",
        source="s",
        timestamp="t",
    )
    assert cib.detect_ai_signals(up)


def test_process_roundtrip(tmp_path, monkeypatch):
    sample = [
        cib.CompetitorUpdate(
            title="Launch",
            content="Great profit with AI tools",
            source="x",
            timestamp="t",
        )
    ]
    monkeypatch.setattr(cib, "fetch_updates", lambda url: sample)
    db = cib.IntelligenceDB(tmp_path / "int.db")
    bot = cib.CompetitiveIntelligenceBot(db)
    res = bot.process(["http://example.com"])
    assert res and res[0].sentiment != 0
    stored = db.fetch()
    assert stored and stored[0].ai_signals
