import menace.whisper_utils as wu
import types


def test_missing_deps_warns(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(wu, "whisper", None)
    monkeypatch.setattr(wu, "sr", None)
    result = wu.transcribe_with_whisper("a.wav")
    assert result is None
    assert "unavailable" in caplog.text


def test_failed_transcription_returns_none(monkeypatch, tmp_path, caplog):
    caplog.set_level("WARNING")

    class DummyModel:
        def transcribe(self, path):
            raise RuntimeError("fail")

    monkeypatch.setattr(
        wu,
        "whisper",
        types.SimpleNamespace(load_model=lambda name: DummyModel()),
    )
    monkeypatch.setattr(wu, "sr", None)

    result = wu.transcribe_with_whisper(str(tmp_path / "a.wav"))
    assert result is None
    assert "falling back" in caplog.text


def test_online_fallback(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(wu, "whisper", None)
    monkeypatch.setattr(wu, "sr", None)

    called = {}

    def fake_online(p, k):
        called["ok"] = True
        return "online"

    monkeypatch.setattr(wu, "_online_transcribe", fake_online)
    monkeypatch.setattr(wu, "_lightweight_transcribe", lambda p: None)

    result = wu.transcribe_with_whisper(str(tmp_path / "b.wav"), api_key="k")
    assert result == "online"
    assert called.get("ok")
    assert "online service" in caplog.text


def test_local_fallback(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(wu, "whisper", None)
    monkeypatch.setattr(wu, "sr", None)

    monkeypatch.setattr(wu, "_online_transcribe", lambda p, k: None)
    monkeypatch.setattr(wu, "_lightweight_transcribe", lambda p: "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = wu.transcribe_with_whisper(str(tmp_path / "c.wav"))
    assert result == "local"
    assert "lightweight fallback" in caplog.text
