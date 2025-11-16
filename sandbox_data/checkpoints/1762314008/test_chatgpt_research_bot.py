import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.chatgpt_research_bot as crb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402
from menace.llm_interface import LLMResult  # noqa: E402


def test_summarise_text():
    text = "One. Two. Three."
    summary = crb.summarise_text(text, ratio=0.34)
    assert "One" in summary
    assert summary.count(".") <= 2


def test_process(monkeypatch):
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""
    builder = DummyBuilder()
    client = cib.ChatGPTClient(api_key="key", context_builder=builder)
    responses = [
        LLMResult(raw={"choices": [{"message": {"content": "Answer one."}}]}, text="Answer one."),
        LLMResult(raw={"choices": [{"message": {"content": "Answer two."}}]}, text="Answer two."),
    ]
    count = 0

    def fake_generate(prompt, **kw):
        nonlocal count
        resp = responses[count]
        count += 1
        return resp

    monkeypatch.setattr(client, "generate", fake_generate)
    sent = {}

    def fake_send(conv, summary):
        sent["conv"] = conv
        sent["summary"] = summary

    monkeypatch.setattr(crb, "send_to_aggregator", fake_send)
    bot = crb.ChatGPTResearchBot(builder, client)
    result = bot.process("Question", depth=2, ratio=0.5)
    assert len(result.conversation) == 2
    assert sent["conv"] == result.conversation
    assert result.summary
