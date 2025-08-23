import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from pathlib import Path
from types import SimpleNamespace
import pytest

import menace.video_research_bot as vrb


def test_summarise_text():
    text = "Sentence one. Sentence two. Sentence three."
    summary = vrb.summarise_text(text, ratio=0.34)
    assert "Sentence" in summary
    assert summary.count(".") <= 2


def test_process(monkeypatch, tmp_path: Path):
    bot = vrb.VideoResearchBot(api_key="test", storage_dir=tmp_path)
    monkeypatch.setattr(bot, "search", lambda q: ["http://youtube.com/watch?v=1"])
    monkeypatch.setattr(bot, "download_video", lambda url: str(tmp_path / "v.mp4"))
    monkeypatch.setattr(bot, "extract_audio", lambda p: str(tmp_path / "a.wav"))
    monkeypatch.setattr(bot, "transcribe", lambda p: "word " * 50)

    results = bot.process("query", ratio=0.5)
    assert results and results[0].url.endswith("1")
    assert results[0].path.endswith("v.mp4")
    assert results[0].audio_path.endswith("a.wav")
    assert results[0].transcript
    assert results[0].summary


def test_whisper_transcribe(monkeypatch, tmp_path: Path):
    if not vrb.whisper_utils.whisper:
        pytest.skip("whisper not available")
    bot = vrb.VideoResearchBot(api_key="k", storage_dir=tmp_path)
    monkeypatch.setattr(vrb.whisper_utils, "transcribe_with_whisper", lambda p: "ok")
    text = bot.transcribe(str(tmp_path / "a.wav"))
    assert text == "ok"


def test_db_hit_skips_network(monkeypatch, tmp_path: Path):
    from menace.research_aggregator_bot import ResearchItem
    info = ResearchItem(topic="T", content="trans", summary="sum", timestamp=0.0, source_url="http://y")
    router = SimpleNamespace()
    router.query_all = lambda q: SimpleNamespace(code=[], bots=[], info=[info], memory=[], menace=[])
    bot = vrb.VideoResearchBot(api_key="k", storage_dir=tmp_path, db_router=router)

    def fail(*a, **k):
        raise AssertionError("network")

    monkeypatch.setattr(bot, "search", fail)
    monkeypatch.setattr(bot, "download_video", fail)
    results = bot.process("T")
    assert results and results[0].summary == "sum"
