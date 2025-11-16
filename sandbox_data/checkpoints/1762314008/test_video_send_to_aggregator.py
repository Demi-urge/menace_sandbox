import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("menace.db_router", types.SimpleNamespace(DBRouter=object))
sys.modules.setdefault("pytube", types.ModuleType("pytube"))
sys.modules["pytube"].YouTube = object
sys.modules.setdefault("speech_recognition", types.ModuleType("sr"))
sys.modules["speech_recognition"].Recognizer = object
sys.modules["speech_recognition"].AudioFile = object
sys.modules["speech_recognition"].UnknownValueError = Exception
import menace.video_research_bot as vrb


def _video_item():
    return vrb.VideoItem(url="u", transcript="t", summary="s", path="p", audio_path="a")


def test_requests_error_logs_warning(monkeypatch, caplog):
    caplog.set_level("WARNING")

    def fake_post(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr(vrb, "requests", types.SimpleNamespace(post=fake_post))
    ok = vrb.send_to_aggregator([_video_item()])
    assert not ok
    assert "Failed to send videos to aggregator" in caplog.text


def test_urllib_error_logs_warning(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(vrb, "requests", None)

    def fake_urlopen(*a, **k):
        raise vrb.urlerror.URLError("fail")

    monkeypatch.setattr(vrb.urlrequest, "urlopen", fake_urlopen)
    ok = vrb.send_to_aggregator([_video_item()])
    assert not ok
    assert "Failed to send videos to aggregator" in caplog.text

