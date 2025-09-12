from menace.coding_bot_interface import self_coding_managed


@self_coding_managed
class DummyCodingBot:
    """Minimal bot used for self-coding pipeline tests."""

    name = "dummy_coding_bot"

    def __init__(self, **_kwargs: object) -> None:  # pragma: no cover - simple
        """Accept arbitrary kwargs so the decorator can inject deps."""
        pass
