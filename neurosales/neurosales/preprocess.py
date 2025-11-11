from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List
from .metrics import metrics

import emoji
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional heavy deps
    spacy = None


DEFAULT_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "you're": "you are",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "let's": "let us",
}

SLANG_MAP = {
    "u": "you",
    "r": "are",
    "pls": "please",
    "thx": "thanks",
}

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+")


@dataclass
class PreprocessResult:
    tokens: List[str]
    lemmas: List[str]
    trigger_flags: Dict[str, int]
    urls: List[str]
    emails: List[str]


class TextPreprocessor:
    """Fast text preprocessing for trigger analysis."""

    def __init__(self, trigger_words: Iterable[str]) -> None:
        self.trigger_words = {w.lower() for w in trigger_words}
        if spacy is not None:
            print("[DEBUG] Current PATH during spacy load:", os.environ["PATH"])
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        else:  # pragma: no cover - fallback
            self.nlp = None

    def _expand_contractions(self, text: str) -> str:
        for c, e in DEFAULT_CONTRACTIONS.items():
            text = re.sub(r"\b" + re.escape(c) + r"\b", e, text)
        return text

    def _convert_slang(self, tok: str) -> str:
        return SLANG_MAP.get(tok, tok)

    def preprocess(self, text: str) -> PreprocessResult:
        text = text.lower()
        text = self._expand_contractions(text)
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = re.sub(r"[^\w\s@:/.]", " ", text)
        urls = URL_RE.findall(text)
        emails = EMAIL_RE.findall(text)
        tokens: List[str] = []
        lemmas: List[str] = []
        trigger_flags: Dict[str, int] = {}
        if self.nlp is not None:
            doc = self.nlp(text)
            for token in doc:
                if token.is_space:
                    continue
                word = self._convert_slang(token.text)
                lemma = token.lemma_.lower()
                lemma = self._convert_slang(lemma)
                tokens.append(word)
                lemmas.append(lemma)
                for trg in self.trigger_words:
                    if lemma == trg or trg in lemma:
                        trigger_flags[trg] = trigger_flags.get(trg, 0) + 1
        else:  # fallback simple splitting
            for word in text.split():
                word = self._convert_slang(word)
                lemma = word
                if lemma.endswith("ing") and len(lemma) > 4:
                    lemma = lemma[:-3]
                tokens.append(word)
                lemmas.append(lemma)
                for trg in self.trigger_words:
                    if lemma == trg or trg in lemma:
                        trigger_flags[trg] = trigger_flags.get(trg, 0) + 1
        metrics.record_tokens(len(tokens))
        return PreprocessResult(tokens, lemmas, trigger_flags, urls, emails)
