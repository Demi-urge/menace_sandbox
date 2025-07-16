import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.personalized_conversation as pc


class StubClient:
    def ask(self, msgs):
        return {"choices": [{"message": {"content": "hello!"}}]}


def test_modes():
    bot = pc.PersonalizedConversationManager(StubClient(), mode="casual")
    text = bot.ask("hi")
    assert ":)" in text
    bot.set_mode("formal")
    text = bot.ask("hi")
    assert not text.endswith("!)")
