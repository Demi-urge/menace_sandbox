"""Utilities for topic classification of clips."""

from __future__ import annotations

import argparse
import json
import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    import subprocess
except Exception:  # pragma: no cover - subprocess always available
    subprocess = None

from .. import whisper_utils

logger = logging.getLogger(__name__)


def _extract_text(path: Path) -> str:
    """Return transcript text for *path* if possible."""
    txt = path.with_suffix(".txt")
    if txt.exists():
        try:
            return txt.read_text(encoding="utf-8")
        except Exception:
            return ""
    if subprocess is None:
        return ""
    audio_path = path.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        text = whisper_utils.transcribe_with_whisper(str(audio_path))
        return text or ""
    except Exception:
        return ""


def _score(text: str, keywords: List[str]) -> float:
    text = text.lower()
    score = 0.0
    for kw in keywords:
        kw_low = kw.lower()
        if kw_low:
            score += text.count(kw_low)
    return score


def process_clip(path: Path, topics: Dict[str, Any]) -> None:
    """Classify a single clip into one of *topics* and store results."""
    text = _extract_text(path)
    if not text:
        text = path.stem

    best_topic = None
    best_score = 0.0
    for name, data in topics.items():
        words = data.get("keywords", [name])
        if isinstance(words, str):
            words = [words]
        score = _score(text, words)
        if score > best_score:
            best_topic = name
            best_score = score

    if best_topic is None:
        best_topic = "unknown"
    entry = topics.setdefault(best_topic, {})
    entry["clip_count"] = entry.get("clip_count", 0) + 1

    result = {
        "file": path.name,
        "topic": best_topic,
        "confidence": float(best_score),
    }
    out_file = path.with_suffix(".json")
    for i in range(3):
        try:
            with out_file.open("w", encoding="utf-8") as fh:
                json.dump(result, fh)
            break
        except Exception as exc:  # pragma: no cover - I/O failures
            logger.warning("Failed to write classification for %s: %s", path, exc)
            time.sleep(0.1)


def classify_clips(input_dir: str, topics_file: str) -> None:
    os.makedirs(input_dir, exist_ok=True)
    with open(topics_file, "r", encoding="utf-8") as fh:
        topics = json.load(fh)
    for name in os.listdir(input_dir):
        if name.endswith(".mp4"):
            process_clip(Path(input_dir) / name, topics)


def cli(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output_clips")
    parser.add_argument("--topics", default="clip_topics.json")
    args = parser.parse_args(argv)
    classify_clips(args.input, args.topics)


__all__ = ["classify_clips", "process_clip", "cli"]
