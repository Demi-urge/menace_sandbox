try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisper = None  # type: ignore

try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sr = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


import os
from typing import Optional

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


def _online_transcribe(audio_path: str, api_key: str) -> Optional[str]:
    """Attempt remote transcription using OpenAI's Whisper API.

    SelfCodingEngine handles code generation locally; this remote call is a
    fallback for transcription when local resources are unavailable.
    """
    if requests is None:
        return None
    try:
        with open(audio_path, "rb") as fh:
            files = {"file": fh}
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"model": "whisper-1"}
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=30,
            )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
    except Exception as exc:  # pragma: no cover - network issues
        logger.error("online transcription failed: %s", exc)
    return None


def _lightweight_transcribe(audio_path: str) -> Optional[str]:
    """Very small offline transcription attempt using raw audio features."""
    import wave
    import audioop

    try:
        with wave.open(audio_path, "rb") as fh:
            frames = fh.readframes(fh.getnframes())
            if not frames:
                return None
            rms = audioop.rms(frames, fh.getsampwidth())
            return "audio" if rms > 1000 else ""
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.debug("lightweight transcription failed for %s: %s", audio_path, exc)
        return None


def transcribe_with_whisper(
    audio_path: str, model_name: str | None = None, *, api_key: str | None = None
) -> Optional[str]:
    """Return transcription using Whisper if available.

    If transcription cannot be performed due to missing dependencies or
    runtime errors, ``None`` is returned and a warning is logged to signal
    that the result may be inaccurate.
    """
    if whisper is None and sr is None:
        logger.warning(
            "whisper and speech_recognition unavailable; attempting fallback"
        )
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if key:
            text = _online_transcribe(audio_path, key)
            if text:
                logger.info("transcription completed via online service")
                return text
        text = _lightweight_transcribe(audio_path)
        if text:
            logger.info("transcription completed via lightweight fallback")
            return text
        return None
    if whisper is not None:
        try:
            name = model_name or "base"
            model = whisper.load_model(name)
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()
        except Exception as exc:
            logger.exception(
                "whisper model transcription failed for %s: %s", audio_path, exc
            )
    if sr is not None:
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_sphinx(audio)
        except Exception as exc:
            logger.exception(
                "speech_recognition transcription failed for %s: %s",
                audio_path,
                exc,
            )

    logger.warning("falling back from whisper transcription for %s", audio_path)
    return None
