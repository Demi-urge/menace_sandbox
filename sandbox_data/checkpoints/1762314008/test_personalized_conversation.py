import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.personalized_conversation as pc
from prompt_types import Prompt


class StubBuilder:
    def refresh_db_weights(self):
        pass

    def build_prompt(self, query, **_):
        return Prompt(user=query)


class StubClient:
    context_builder = StubBuilder()

    def ask(self, prompt_obj, **_):
        assert isinstance(prompt_obj, Prompt)
        return {"choices": [{"message": {"content": "hello!"}}]}


def test_modes():
    bot = pc.PersonalizedConversationManager(StubClient(), mode="casual")
    text = bot.ask("hi")
    assert ":)" in text
    bot.set_mode("formal")
    text = bot.ask("hi")
    assert not text.endswith("!)")


def test_resistance_triggers_strategy_and_cta():
    bot = pc.PersonalizedConversationManager(StubClient(), mode="casual")
    bot.ask("no thanks")
    assert bot.mode == "casual"
    assert bot.emotional_strategy == "empathetic"
    assert bot.memory.current_chain() == []
    bot.ask("still not interested")
    assert bot.mode == "formal"
    assert bot.emotional_strategy == "assertive"
    assert bot.memory.current_chain()
