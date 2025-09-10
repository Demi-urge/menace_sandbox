"""Utility functions for text preprocessing before embedding.

This module exposes :func:`generalise` which performs language-aware text
normalisation.  It supports configurable stop word lists, optional language
detection and lemmatisation/stemming.  Databases may register their own
pre-processing configuration which will be used when ``generalise`` is invoked
with a ``db_key``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

# A tiny English stop word list.  This keeps the implementation light and
# avoids pulling in heavy dependencies for the common case.
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

try:  # pragma: no cover - optional dependency
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk.corpus import wordnet  # type: ignore  # noqa: F401

    _LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover - nltk or data missing
    _LEMMATIZER = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from nltk.stem import SnowballStemmer  # type: ignore

    _STEMMER_FACTORY = SnowballStemmer
except Exception:  # pragma: no cover - nltk missing
    _STEMMER_FACTORY = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from langdetect import detect  # type: ignore

    _LANG_DETECT = detect
except Exception:  # pragma: no cover - dependency missing
    _LANG_DETECT = None  # type: ignore


_LANG_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
}


@dataclass
class PreprocessingConfig:
    """Configuration controlling text normalisation."""

    stop_words: Optional[Set[str]] = None
    language: Optional[str] = None
    use_lemmatizer: bool = True


def load_stop_words(source: str | Iterable[str]) -> Set[str]:
    """Return a set of stop words from ``source``.

    ``source`` may be an iterable, a path to a file containing one word per
    line, or a spaCy language code (when spaCy is available).  The function
    gracefully falls back to an empty set when a source cannot be resolved.
    """

    if isinstance(source, str):
        if os.path.exists(source):
            with open(source, "r", encoding="utf8") as fh:
                return {line.strip() for line in fh if line.strip()}
        try:  # pragma: no cover - spaCy is optional
            import spacy  # type: ignore

            nlp = spacy.blank(source)
            return set(nlp.Defaults.stop_words)
        except Exception:
            return set()
    return {w.strip() for w in source if isinstance(w, str) and w.strip()}


_CONFIGS: Dict[str, PreprocessingConfig] = {}


def register_preprocessor(db_key: str, config: PreprocessingConfig) -> None:
    """Register a :class:`PreprocessingConfig` for ``db_key``."""

    _CONFIGS[db_key] = config


def _resolve_config(
    db_key: Optional[str], config: Optional[PreprocessingConfig]
) -> PreprocessingConfig:
    if config is not None:
        return config
    if db_key and db_key in _CONFIGS:
        return _CONFIGS[db_key]
    return PreprocessingConfig()


def _lemmatise(token: str) -> str:
    """Return the lemmatised form of ``token`` if possible."""

    if _LEMMATIZER is None:  # pragma: no cover - fallback path
        return token
    try:  # pragma: no cover - best effort
        return _LEMMATIZER.lemmatize(token)
    except Exception:
        return token


def generalise(
    text: str, *, config: PreprocessingConfig | None = None, db_key: str | None = None
) -> str:
    """Return a condensed representation of ``text``.

    The function performs the following steps:

    * lowercases the input text
    * tokenises on word boundaries
    * removes stop words (configurable)
    * optionally lemmatises or stems tokens depending on the detected or
      configured language

    Parameters
    ----------
    text:
        The input text to normalise.
    config:
        Optional :class:`PreprocessingConfig` describing how ``text`` should be
        processed.  If not supplied, a configuration registered for ``db_key``
        will be used if available.
    db_key:
        Identifier of the database requesting the preprocessing.  This allows
        different databases to specify custom preprocessing behaviour.

    Returns
    -------
    str
        A space separated string of processed tokens.
    """

    if not isinstance(text, str):  # pragma: no cover - defensive
        return text

    cfg = _resolve_config(db_key, config)
    lang = cfg.language
    if lang is None and _LANG_DETECT is not None:
        try:  # pragma: no cover - best effort
            lang = _LANG_DETECT(text)
        except Exception:
            lang = "en"
    lang = lang or "en"

    stop_words = cfg.stop_words or _STOP_WORDS

    tokens = re.findall(r"\b\w+\b", text.lower())
    processed = []

    stemmer = None
    if lang != "en" and _STEMMER_FACTORY is not None and cfg.use_lemmatizer:
        stem_lang = _LANG_MAP.get(lang, lang)
        try:  # pragma: no cover - best effort
            stemmer = _STEMMER_FACTORY(stem_lang)
        except Exception:
            stemmer = None

    for tok in tokens:
        if tok in stop_words:
            continue
        if stemmer is not None:
            try:  # pragma: no cover - best effort
                tok = stemmer.stem(tok)
            except Exception:
                pass
        elif cfg.use_lemmatizer:
            tok = _lemmatise(tok)
        processed.append(tok)
    return " ".join(processed)


__all__ = [
    "PreprocessingConfig",
    "load_stop_words",
    "register_preprocessor",
    "generalise",
]
