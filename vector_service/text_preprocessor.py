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
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Union

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml may be missing
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore
except Exception:  # pragma: no cover - spaCy may be missing
    spacy = None  # type: ignore

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
    "nl": "dutch",
    "sv": "swedish",
    "da": "danish",
    "no": "norwegian",
    "fi": "finnish",
    "ru": "russian",
    "pl": "polish",
}


@dataclass
class PreprocessingConfig:
    """Configuration controlling text normalisation and chunking."""

    stop_words: Optional[Union[Iterable[str], str]] = None
    language: Optional[str] = None
    use_lemmatizer: bool = True
    split_sentences: bool = True
    chunk_size: int = 400
    filter_semantic_risks: bool = True


def load_stop_words(source: str | Iterable[str]) -> Set[str]:
    """Return a set of stop words from ``source``.

    ``source`` may be an iterable, a path to a file containing one word per
    line, or a language code recognised by spaCy or NLTK.  The function
    gracefully falls back to an empty set when a source cannot be resolved.
    """

    if isinstance(source, str):
        if os.path.exists(source):
            with open(source, "r", encoding="utf8") as fh:
                return {line.strip() for line in fh if line.strip()}
        try:  # pragma: no cover - NLTK is optional
            from nltk.corpus import stopwords  # type: ignore

            lang = _LANG_MAP.get(source, source)
            return set(stopwords.words(lang))
        except Exception:
            pass
        try:  # pragma: no cover - spaCy is optional
            if spacy is not None:
                nlp = spacy.blank(source)
                return set(nlp.Defaults.stop_words)
        except Exception:
            pass
        return set()
    return {w.strip() for w in source if isinstance(w, str) and w.strip()}


_CONFIGS: Dict[str, PreprocessingConfig] = {}


def register_preprocessor(db_key: str, config: PreprocessingConfig) -> None:
    """Register a :class:`PreprocessingConfig` for ``db_key``."""

    _CONFIGS[db_key] = config


def get_config(db_key: str) -> PreprocessingConfig:
    """Return the :class:`PreprocessingConfig` for ``db_key`` if registered."""

    return _CONFIGS.get(db_key, PreprocessingConfig())


def load_db_configs(path: str) -> None:
    """Load preprocessing configs from JSON or YAML ``path``."""

    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf8") as fh:
        if path.endswith((".yml", ".yaml")) and yaml is not None:
            data = yaml.safe_load(fh) or {}
        else:
            data = json.load(fh)
    if not isinstance(data, dict):
        return
    for key, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        stop = cfg.get("stop_words")
        if isinstance(stop, (list, set, tuple)):
            stop_words = set(stop)
        else:
            stop_words = stop
        register_preprocessor(
            key,
            PreprocessingConfig(
                stop_words=stop_words,
                language=cfg.get("language"),
                use_lemmatizer=cfg.get("use_lemmatizer", True),
                split_sentences=cfg.get("split_sentences", True),
                chunk_size=int(cfg.get("chunk_size", 400)),
                filter_semantic_risks=cfg.get("filter_semantic_risks", True),
            ),
        )


# Load default preprocessing configs from the packaged ``preprocess.yml`` at import
# time so databases automatically pick up their settings without manual
# registration.
load_db_configs(os.path.join(os.path.dirname(__file__), "preprocess.yml"))


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

    stop_words: Optional[Set[str]]
    sw_source = cfg.stop_words
    if isinstance(sw_source, str):
        stop_words = load_stop_words(sw_source)
    elif sw_source is not None:
        stop_words = {w for w in sw_source if isinstance(w, str)}
    else:
        stop_words = load_stop_words(lang) or _STOP_WORDS

    tokens = None
    if spacy is not None:
        try:  # pragma: no cover - spaCy tokeniser
            tokens = [t.text for t in spacy.blank(lang)(text.lower())]
        except Exception:
            tokens = None
    if tokens is None:
        tokens = re.findall(r"\b\w+\b", text.lower())
    processed: list[str] = []

    stemmer = None
    lemmatizer = _LEMMATIZER if lang == "en" and cfg.use_lemmatizer else None
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
        elif lemmatizer is not None:
            tok = _lemmatise(tok)
        processed.append(tok)
    return " ".join(processed)


__all__ = [
    "PreprocessingConfig",
    "load_stop_words",
    "register_preprocessor",
    "get_config",
    "load_db_configs",
    "generalise",
]
