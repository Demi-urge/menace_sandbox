"""Conversation Manager Bot."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, List

from datetime import datetime, timezone

from neurosales import (
    add_message as mq_add_message,
    get_recent_messages,
    push_chain,
    peek_chain,
    MessageEntry,
    CTAChain,
)

from .report_generation_bot import ReportGenerationBot, ReportOptions

from .chatgpt_idea_bot import ChatGPTClient
from gpt_memory_interface import GPTMemoryInterface
from prompt_types import Prompt
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover - optional
    sr = None  # type: ignore

try:
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover - optional
    gTTS = None  # type: ignore


@dataclass
class ConversationResult:
    """Result of a conversation request."""

    text: str
    audio_path: Optional[Path] = None


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class ConversationManagerBot:
    """Manage queries to ChatGPT and Stage 7 bots with optional speech support."""

    def __init__(
        self,
        client: ChatGPTClient,
        stage7_bots: Optional[Dict[str, Callable[[str], str]]] = None,
        report_bot: ReportGenerationBot | None = None,
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
    ) -> None:
        self.client = client
        self.stage7_bots = stage7_bots or {}
        self.cache: Dict[str, str] = {}
        self.recognizer = sr.Recognizer() if sr else None
        self._notifications: List[str] = []
        self.report_bot = report_bot or ReportGenerationBot()
        self.strategy = "neutral"
        self.gpt_memory = gpt_memory
        if getattr(self.client, "gpt_memory", None) is None:
            try:
                self.client.gpt_memory = self.gpt_memory
            except Exception:
                logger.debug("failed to attach gpt_memory to client", exc_info=True)
        self._objection_keywords = {"no", "not", "don't", "cant", "can't", "won't"}
        self.resistance_handler: Callable[[List[MessageEntry], CTAChain | None], None] | None = None

    def on_resistance(
        self, handler: Callable[[List[MessageEntry], CTAChain | None], None]
    ) -> None:
        """Register a callback for resistance detection events."""

        self.resistance_handler = handler

    def notify(self, message: str) -> None:
        """Add a notification to the queue."""
        self._notifications.append(message)

    def get_notifications(self) -> List[str]:
        notes = list(self._notifications)
        self._notifications.clear()
        return notes

    # ------------------------------------------------------------------
    # Internal helpers

    def _apply_strategy(self, query: str) -> str:
        """Adjust the query based on the current strategy."""

        if self.strategy == "conciliatory":
            return f"Please respond in a calm and understanding tone: {query}"
        if self.strategy == "assertive":
            return f"Provide a firm and direct answer: {query}"
        return query

    def _update_memory(self, user_text: str, agent_text: str) -> None:
        """Store user and agent messages in memory structures."""

        mq_add_message(user_text)
        mq_add_message(agent_text)
        recent = get_recent_messages(2)
        if len(recent) == 2:
            push_chain(recent[0], recent[1], datetime.now(timezone.utc))

    def _is_objection(self, text: str) -> bool:
        lower = text.lower()
        return any(word in lower for word in self._objection_keywords)

    def _detect_resistance(self) -> None:
        """Check recent messages for repeated objections and flip strategy."""

        recent = get_recent_messages(3)
        objections = [m for m in recent if self._is_objection(m.text)]
        if len(objections) >= 2:
            self.strategy = "conciliatory" if self.strategy == "neutral" else "assertive"
            chain = peek_chain()
            if self.resistance_handler:
                self.resistance_handler(objections, chain)

    def _chatgpt(self, prompt: str) -> str:
        if prompt in self.cache:
            return self.cache[prompt]

        intent_meta = {"query": prompt, "tags": [FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT]}
        try:
            prompt_obj = self.client.context_builder.build_prompt(
                prompt, intent_metadata=intent_meta
            )
        except Exception:
            logger.exception("ContextBuilder.build_prompt failed")
            prompt_obj = Prompt(user=prompt)

        parts: List[str] = [prompt_obj.user]
        if getattr(prompt_obj, "examples", None):
            parts.append("\n".join(prompt_obj.examples))

        messages: List[Dict[str, object]] = []
        if getattr(prompt_obj, "system", None):
            messages.append({"role": "system", "content": prompt_obj.system})
        messages.append(
            {
                "role": "user",
                "content": "\n".join(parts),
                "metadata": getattr(prompt_obj, "metadata", {}) or {},
            }
        )

        data = self.client.ask(
            messages,
            tags=intent_meta["tags"],
            memory_manager=self.gpt_memory,
        )
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self._update_memory(prompt_obj.user, text)
        logger.debug("prompt: %s\nresponse: %s", prompt_obj.user, text)
        self.cache[prompt] = text
        return text

    def ask(self, query: str, target_bot: Optional[str] = None) -> str:
        """Return answer from a Stage 7 bot or ChatGPT and update memory."""

        adjusted = self._apply_strategy(query)
        if target_bot and target_bot in self.stage7_bots:
            response = self.stage7_bots[target_bot](adjusted)
            self._update_memory(query, response)
        else:
            response = self._chatgpt(adjusted)
        self._detect_resistance()
        return response

    def request_report(
        self,
        start: str | None = None,
        end: str | None = None,
        metrics: Optional[List[str]] = None,
    ) -> Path:
        """Generate a metrics report for the given period."""
        opts = ReportOptions(metrics=metrics or ["cpu", "memory"])
        report = self.report_bot.compile_report(opts, limit=None, start=start, end=end)
        self.notify(f"Report generated: {report}")
        return report

    # Speech helpers -----------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio to text."""
        if not sr or not self.recognizer:
            raise ImportError("SpeechRecognition is required for transcribe")
        with sr.AudioFile(str(audio_path)) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio)
        except Exception:
            return ""

    def synthesize(self, text: str, out_path: Optional[Path] = None) -> Optional[Path]:
        """Convert text to speech."""
        if not gTTS:
            raise ImportError("gTTS is required for text to speech")
        tts = gTTS(text)
        path = out_path or Path("output.mp3")
        try:
            tts.save(str(path))
            return path
        except Exception:
            return None

    def ask_audio(self, audio_path: Path, target_bot: Optional[str] = None) -> ConversationResult:
        """Process an audio query and return text and optional speech output."""
        text = self.transcribe(audio_path)
        if not text:
            return ConversationResult(text="")
        response = self.ask(text, target_bot)
        audio = self.synthesize(response)
        return ConversationResult(text=response, audio_path=audio)


__all__ = ["ConversationResult", "ConversationManagerBot"]