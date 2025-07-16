import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.chatgpt_research_bot as crb
import menace.chatgpt_idea_bot as cib


def test_summarise_text():
    text = "One. Two. Three."
    summary = crb.summarise_text(text, ratio=0.34)
    assert "One" in summary
    assert summary.count(".") <= 2


def test_process(monkeypatch):
    client = cib.ChatGPTClient("key")
    responses = [
        {"choices": [{"message": {"content": "Answer one."}}]},
        {"choices": [{"message": {"content": "Answer two."}}]},
    ]
    count = 0

    def fake_ask(messages):
        nonlocal count
        resp = responses[count]
        count += 1
        return resp

    monkeypatch.setattr(client, "ask", fake_ask)
    sent = {}

    def fake_send(conv, summary):
        sent["conv"] = conv
        sent["summary"] = summary

    monkeypatch.setattr(crb, "send_to_aggregator", fake_send)
    bot = crb.ChatGPTResearchBot(client)
    result = bot.process("Question", depth=2, ratio=0.5)
    assert len(result.conversation) == 2
    assert sent["conv"] == result.conversation
    assert result.summary
