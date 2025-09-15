import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from pathlib import Path  # noqa: E402

import menace.report_generation_bot as rgb  # noqa: E402
import menace.data_bot as db  # noqa: E402

import menace.conversation_manager_bot as cmb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402


from prompt_types import Prompt


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_prompt(self, query, **_):
        return Prompt(user=query)


def test_cache(monkeypatch):
    client = cib.ChatGPTClient("key", context_builder=DummyBuilder())
    calls = []

    def fake_ask(msgs):
        calls.append(1)
        return {"choices": [{"message": {"content": "reply"}}]}

    monkeypatch.setattr(client, "ask", fake_ask)
    bot = cmb.ConversationManagerBot(client)
    r1 = bot.ask("hello")
    r2 = bot.ask("hello")
    assert r1 == "reply" and r2 == "reply"
    assert len(calls) == 1


def test_audio_flow(monkeypatch, tmp_path: Path):
    client = cib.ChatGPTClient("key", context_builder=DummyBuilder())
    monkeypatch.setattr(
        client,
        "ask",
        lambda msgs: {"choices": [{"message": {"content": "answer"}}]},
    )
    bot = cmb.ConversationManagerBot(
        client, stage7_bots={"data": lambda q: "data"}
    )
    monkeypatch.setattr(bot, "transcribe", lambda p: "what is up")
    monkeypatch.setattr(bot, "synthesize", lambda t, out_path=None: tmp_path / "out.mp3")
    result = bot.ask_audio(Path("foo.wav"), target_bot="data")
    assert result.text == "data"
    assert result.audio_path and result.audio_path.name == "out.mp3"


def test_request_report(tmp_path: Path):
    client = cib.ChatGPTClient("key", context_builder=DummyBuilder())
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=1.0,
            memory=1.0,
            response_time=0.1,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
        )
    )
    reporter = rgb.ReportGenerationBot(db=mdb, reports_dir=tmp_path)
    bot = cmb.ConversationManagerBot(client, report_bot=reporter)
    report = bot.request_report(start="1970-01-01", end="2100-01-01")
    assert report.exists()
    notes = bot.get_notifications()
    assert notes and "Report" in notes[0]
