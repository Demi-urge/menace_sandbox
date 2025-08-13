from __future__ import annotations

"""Conversation manager with casual/formal modes and style mirroring."""

from typing import Optional
import re

from .conversation_manager_bot import ConversationManagerBot
from .chatgpt_idea_bot import ChatGPTClient
from .mirror_bot import MirrorBot
from neurosales.interaction_memory import InteractionMemory


class PersonalizedConversationManager(ConversationManagerBot):
    """Enhance ConversationManagerBot with user style adaptation."""

    def __init__(
        self,
        client: ChatGPTClient,
        mode: str = "formal",
        mirror: MirrorBot | None = None,
        memory: InteractionMemory | None = None,
        **kwargs,
    ) -> None:
        super().__init__(client, **kwargs)
        self.mode = mode
        self.mirror = mirror or MirrorBot()
        self.memory = memory or InteractionMemory()
        self._objection_keywords = {"no", "not", "don't", "cant", "can't", "won't"}
        self.emotional_strategy = "empathetic"

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

    # --- Resistance detection -----------------------------------
    def detect_resistance(self) -> None:
        """Detect repeated objections and adjust strategy."""

        recent = self.memory.recent_messages()
        negatives = [
            m
            for m in recent
            if m.role == "user"
            and any(k in m.message.lower() for k in self._objection_keywords)
        ]
        if len(negatives) >= 2:
            if self.mode == "casual":
                self.mode = "formal"
            else:
                self.notify("resistance detected")
            self.emotional_strategy = (
                "assertive"
                if self.emotional_strategy == "empathetic"
                else "empathetic"
            )
            self.memory.push_event(
                "resistance",
                escalation_level=len(self.memory.current_chain()) + 1,
            )

    def ask(self, query: str, target_bot: Optional[str] = None) -> str:  # type: ignore[override]
        adjusted = self._apply_strategy(query)
        if target_bot and target_bot in self.stage7_bots:
            response = self.stage7_bots[target_bot](adjusted)
        else:
            response = self._chatgpt(adjusted)

        # Record conversation without personal identifiers
        self.memory.append_message(query, "user")
        self.memory.append_message(response, "assistant")

        self.detect_resistance()

        self.mirror.log_interaction(query, response)

        style_hint = self._derive_style(query)
        if style_hint:
            self.mirror.update_style(style_hint)

        response = self.mirror.generate_response(response)

        if self.mode == "casual":
            return self._to_casual(response)
        return self._to_formal(response)


__all__ = ["PersonalizedConversationManager"]
