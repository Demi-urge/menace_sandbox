"""Generate simple usernames based on topics."""

from __future__ import annotations

import random
import re
import string
import logging
import json
from pathlib import Path
from typing import Iterable, Set, List
from urllib.request import urlopen

from dynamic_path_router import resolve_path


try:
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    wn = None  # type: ignore

import nltk

# Download only if not already present
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

logger = logging.getLogger(__name__)

_DEFAULT_ADJECTIVES = [
    "fast",
    "smart",
    "dynamic",
    "stealthy",
    "bright",
    "clever",
    "bold",
    "silent",
]

_ADJ_SOURCE_URL = "https://raw.githubusercontent.com/dariusk/corpora/master/data/words/adjs.json"


def _fetch_online_adjectives() -> Set[str]:
    """Fetch adjectives from a remote JSON list."""
    try:
        with urlopen(_ADJ_SOURCE_URL, timeout=5) as resp:
            data = json.load(resp)
            return {w.lower() for w in data.get("adjs", []) if w.isalpha()}
    except Exception as exc:  # pragma: no cover - network failures
        logger.exception("failed to fetch adjectives from %s: %s", _ADJ_SOURCE_URL, exc)
        return set()

_ADJ_FILE = resolve_path("adjectives.txt")


def _load_adjectives() -> List[str]:
    """Return adjectives from WordNet, an online source, or bundled file."""
    words: Set[str] = set()
    if wn is not None:
        try:
            for base in _DEFAULT_ADJECTIVES:
                for syn in wn.synsets(base):
                    for lemma in syn.lemma_names():
                        cleaned = lemma.replace("_", "").lower()
                        if cleaned.isalpha():
                            words.add(cleaned)
        except LookupError:
            logger.info(
                "nltk wordnet corpus not found; run 'python -m nltk.downloader wordnet'"
            )
        except Exception as exc:  # pragma: no cover - optional dependency failures
            logger.debug("wordnet adjective lookup failed: %s", exc)
    if not words:
        words.update(_fetch_online_adjectives())
    if not words:
        try:
            adj_path = Path(str(_ADJ_FILE))
            if adj_path.exists():
                with open(str(adj_path), encoding="utf-8") as fh:
                    words.update({w.strip() for w in fh if w.strip()})
        except Exception as exc:  # pragma: no cover - file access failures
            logger.exception("failed to load adjectives: %s", exc)
    if not words:
        words.update(_DEFAULT_ADJECTIVES)
    return sorted(words)


def _get_topic_words(topic: str) -> Iterable[str]:
    words: Set[str] = set()
    for base in re.findall(r"\w+", topic.lower()):
        words.add(base)
        if wn is not None:
            try:
                syns = wn.synsets(base)
                for syn in syns[:3]:
                    for lemma in syn.lemma_names():
                        cleaned = lemma.replace("_", "").lower()
                        if len(cleaned) >= 3:
                            words.add(cleaned)
            except LookupError:
                logger.info(
                    "nltk wordnet corpus not found; run 'python -m nltk.downloader wordnet'"
                )
            except Exception as exc:
                logger.debug("synonym lookup failed for '%s': %s", base, exc)
    return words


def generate_username_for_topic(topic: str, existing: Set[str] | None = None) -> str:
    """Return a unique username related to *topic*.

    The function now uses a richer adjective list and adds a short random
    alphanumeric string rather than a plain integer to better avoid collisions.
    """

    existing = set(existing or [])
    words = list(_get_topic_words(topic)) or ["user"]
    adjectives = _load_adjectives()

    attempts = 0
    while attempts < 1000:
        word = random.choice(words)
        adj = random.choice(adjectives)
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
        name = f"{adj}_{word}_{suffix}"
        if name not in existing:
            return name
        attempts += 1
    raise RuntimeError("unable to generate unique username")

__all__ = ["generate_username_for_topic", "_get_topic_words"]
