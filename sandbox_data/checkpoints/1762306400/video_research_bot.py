from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import os
import json
import subprocess
from dataclasses import dataclass
from typing import List
import os
import logging
from urllib import request as urlrequest, error as urlerror

from .db_router import DBRouter

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    from gensim.summarization import summarize  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    summarize = None  # type: ignore
from pytube import YouTube
import speech_recognition as sr
from . import whisper_utils


def summarise_text(text: str, ratio: float = 0.2) -> str:
    """Return a short summary of the given text."""
    text = text.strip()
    if not text:
        return ""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) <= 1:
        return text
    if summarize:
        try:
            result = summarize(text, ratio=ratio)
            if result:
                return result
        except Exception:
            logging.getLogger(__name__).exception("summarize failed")
    count = max(1, int(len(sentences) * ratio))
    return ". ".join(sentences[:count]) + "."


@dataclass
class VideoItem:
    url: str
    transcript: str
    summary: str
    path: str
    audio_path: str


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class VideoResearchBot:
    """Bot that collects and summarises useful video content."""

    def __init__(self, api_key: str | None = None, storage_dir: str = "videos", db_router: DBRouter | None = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.db_router = db_router

    def search(self, query: str, max_results: int = 1) -> List[str]:
        if self.db_router:
            try:
                hits = self.db_router.query_all(query).info
            except Exception:
                hits = []
            urls = [h.source_url for h in hits if h.source_url]
            if urls:
                return urls[:max_results]
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": self.api_key,
        }
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search", params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in data.get("items", [])
        ]

    def download_video(self, url: str) -> str:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
        return stream.download(self.storage_dir)

    def extract_audio(self, video_path: str) -> str:
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return audio_path

    def transcribe(self, audio_path: str) -> str:
        text = whisper_utils.transcribe_with_whisper(audio_path)
        if text:
            return text
        if text is None or sr is None:
            return ""
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return ""

    def process(self, query: str, ratio: float = 0.2) -> List[VideoItem]:
        if self.db_router:
            try:
                hits = self.db_router.query_all(query).info
            except Exception:
                hits = []
            if hits:
                return [
                    VideoItem(
                        url=h.source_url or "",
                        transcript=h.content or "",
                        summary=h.summary or summarise_text(h.content or "", ratio=ratio),
                        path="",
                        audio_path="",
                    )
                    for h in hits
                ]
        urls = self.search(query)
        results: List[VideoItem] = []
        for url in urls:
            video_path = self.download_video(url)
            audio_path = self.extract_audio(video_path)
            transcript = self.transcribe(audio_path)
            summary = summarise_text(transcript, ratio=ratio)
            results.append(
                VideoItem(
                    url=url,
                    transcript=transcript,
                    summary=summary,
                    path=video_path,
                    audio_path=audio_path,
                )
            )
        return results


def send_to_aggregator(items: List[VideoItem]) -> bool:
    """POST ``items`` to the Research Aggregator service.

    Returns ``True`` if the request succeeded, ``False`` otherwise.
    """

    url = os.getenv("AGGREGATOR_URL", "http://localhost:8000/aggregate")
    payload = [item.__dict__ for item in items]

    if requests:
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover - network issues
            logging.getLogger(__name__).warning(
                "Failed to send videos to aggregator: %s", exc
            )
            return False
    # Fallback to urllib when requests is unavailable
    data = json.dumps(payload).encode()
    req = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:  # pragma: no cover - network
            if 200 <= resp.getcode() < 300:
                return True
            logging.getLogger(__name__).warning(
                "Aggregator returned status %s", resp.getcode()
            )
            return False
    except urlerror.URLError as exc:  # pragma: no cover - network
        logging.getLogger(__name__).warning(
            "Failed to send videos to aggregator: %s", exc
        )
        return False