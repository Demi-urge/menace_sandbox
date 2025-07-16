"""Conversation Manager Bot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, List

from .report_generation_bot import ReportGenerationBot, ReportOptions

from .chatgpt_idea_bot import ChatGPTClient

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


class ConversationManagerBot:
    """Manage queries to ChatGPT and Stage 7 bots with optional speech support."""

    def __init__(
        self,
        client: ChatGPTClient,
        stage7_bots: Optional[Dict[str, Callable[[str], str]]] = None,
        report_bot: ReportGenerationBot | None = None,
    ) -> None:
        self.client = client
        self.stage7_bots = stage7_bots or {}
        self.cache: Dict[str, str] = {}
        self.recognizer = sr.Recognizer() if sr else None
        self._notifications: List[str] = []
        self.report_bot = report_bot or ReportGenerationBot()

    def notify(self, message: str) -> None:
        """Add a notification to the queue."""
        self._notifications.append(message)

    def get_notifications(self) -> List[str]:
        notes = list(self._notifications)
        self._notifications.clear()
        return notes

    def _chatgpt(self, prompt: str) -> str:
        if prompt in self.cache:
            return self.cache[prompt]
        data = self.client.ask([{"role": "user", "content": prompt}])
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self.cache[prompt] = text
        return text

    def ask(self, query: str, target_bot: Optional[str] = None) -> str:
        """Return answer from a Stage 7 bot or ChatGPT."""
        if target_bot and target_bot in self.stage7_bots:
            return self.stage7_bots[target_bot](query)
        return self._chatgpt(query)

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
