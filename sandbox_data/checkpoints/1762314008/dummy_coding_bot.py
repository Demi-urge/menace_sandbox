from menace.coding_bot_interface import self_coding_managed
from menace.bot_registry import BotRegistry
from menace.data_bot import DataBot


registry = BotRegistry()
data_bot = DataBot(start_server=False)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class DummyCodingBot:
    """Minimal bot used for self-coding pipeline tests."""

    name = "dummy_coding_bot"

    def __init__(self, **_kwargs: object) -> None:  # pragma: no cover - simple
        """Accept arbitrary kwargs so the decorator can inject deps."""
        pass
