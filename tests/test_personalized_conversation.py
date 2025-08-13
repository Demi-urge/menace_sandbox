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
