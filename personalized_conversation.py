from __future__ import annotations

"""Conversation manager with casual/formal modes and style mirroring."""

from typing import Optional
import re

from .conversation_manager_bot import ConversationManagerBot
from .chatgpt_idea_bot import ChatGPTClient
from .mirror_bot import MirrorBot


class PersonalizedConversationManager(ConversationManagerBot):
    """Enhance ConversationManagerBot with user style adaptation."""

    def __init__(
        self,
        client: ChatGPTClient,
        mode: str = "formal",
        mirror: MirrorBot | None = None,
        **kwargs,
    ) -> None:
        super().__init__(client, **kwargs)
        self.mode = mode
        self.mirror = mirror or MirrorBot()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    # --- Helpers -------------------------------------------------
    def _derive_style(self, text: str) -> str:
        """Generate a simple style hint from a user message."""
        punct = "!" if "!" in text else ""
        if text.endswith("?"):
            punct += "?"
        return punct

    def _to_casual(self, text: str) -> str:
        text = text.replace("do not", "don't").replace("cannot", "can't")
        if not text.endswith("!"):
            text += "!"
        return f"{text} :)"

    def _to_formal(self, text: str) -> str:
        text = re.sub(r"!+", ".", text)
        if not text.endswith("."):
            text += "."
        return text

    def ask(self, query: str, target_bot: Optional[str] = None) -> str:  # type: ignore[override]
        response = super().ask(query, target_bot)
        self.mirror.log_interaction(query, response)

        style_hint = self._derive_style(query)
        if style_hint:
            self.mirror.update_style(style_hint)

        response = self.mirror.generate_response(response)

        if self.mode == "casual":
            return self._to_casual(response)
        return self._to_formal(response)


__all__ = ["PersonalizedConversationManager"]
