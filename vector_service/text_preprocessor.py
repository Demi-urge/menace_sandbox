"""Utility functions for text preprocessing before embedding.

This module exposes :func:`generalise` which performs basic text
normalisation such as lowercasing, stop word removal and optional
lemmatisation.  The lemmatisation step is attempted if NLTK is available
with the required corpora; otherwise the function gracefully degrades to
simple stop word removal.
"""

from __future__ import annotations

import re
from typing import Iterable

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


def _lemmatise(token: str) -> str:
    """Return the lemmatised form of ``token`` if possible."""

    if _LEMMATIZER is None:  # pragma: no cover - fallback path
        return token
    try:  # pragma: no cover - best effort
        return _LEMMATIZER.lemmatize(token)
    except Exception:
        return token


def generalise(text: str) -> str:
    """Return a condensed representation of ``text``.

    The function performs the following steps:

    * lowercases the input text
    * tokenises on word boundaries
    * removes common English stop words
    * optionally lemmatises tokens when NLTK and its WordNet data are
      available

    Parameters
    ----------
    text:
        The input text to normalise.

    Returns
    -------
    str
        A space separated string of processed tokens.
    """

    if not isinstance(text, str):  # pragma: no cover - defensive
        return text

    tokens = re.findall(r"\b\w+\b", text.lower())
    processed = []
    for tok in tokens:
        if tok in _STOP_WORDS:
            continue
        tok = _lemmatise(tok)
        processed.append(tok)
    return " ".join(processed)


__all__ = ["generalise"]
