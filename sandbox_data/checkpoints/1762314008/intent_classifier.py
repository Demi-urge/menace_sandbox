from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
from collections import defaultdict
import logging

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - heavy optional dep
    SentenceTransformer = None  # type: ignore

try:
    import tensorflow_hub as hub  # type: ignore
except Exception:  # pragma: no cover - heavy optional dep
    hub = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from analysis.semantic_diff_filter import find_semantic_risks
from governed_embeddings import governed_embed
try:
    from compliance.license_fingerprint import check as license_check
except Exception:  # pragma: no cover - optional dependency
    def license_check(text: str):  # type: ignore
        return None
try:
    from security.secret_redactor import redact
except Exception:  # pragma: no cover - optional dependency
    def redact(text: str):  # type: ignore
        return text

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Multi-label intent classifier with optional embedding backends."""

    def __init__(
        self,
        *,
        context_size: int = 3,
        threshold: float = 0.5,
        backend: str = "tfidf",
    ) -> None:
        self.context_size = context_size
        self.threshold = threshold
        self.backend = backend
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: OneVsRestClassifier | None = None
        self.mlb = MultiLabelBinarizer()
        self.trend_bias: Dict[str, float] = defaultdict(lambda: 1.0)
        self._load_backends()

    def _load_backends(self) -> None:
        self._sbert = None
        self._use = None
        if self.backend == "sbert" and SentenceTransformer is not None:
            try:
                from huggingface_hub import login
                import os

                login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
                self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.backend = "tfidf"
        if self.backend == "use" and hub is not None:
            try:
                self._use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            except Exception:
                self.backend = "tfidf"
        if self.backend == "tfidf" or (self._sbert is None and self._use is None):
            self.vectorizer = TfidfVectorizer(stop_words="english")

    def _sanitize_texts(self, texts: Sequence[str]) -> Tuple[List[str], List[int]]:
        clean: List[str] = []
        skipped: List[int] = []
        for idx, t in enumerate(texts):
            red = redact(t.strip())
            if not red:
                skipped.append(idx)
                continue
            lic = license_check(red)
            if lic:
                logger.warning("license detected: %s", lic)
                skipped.append(idx)
                continue
            alerts = find_semantic_risks(red.splitlines())
            if alerts:
                logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
                skipped.append(idx)
                continue
            clean.append(red)
        return clean, skipped

    def _embed(self, texts: Sequence[str]) -> Tuple[Any, List[int]]:
        texts, skipped = self._sanitize_texts(texts)
        if self._sbert is not None:
            vectors = []
            for t in texts:
                vec = governed_embed(t, self._sbert)
                if vec is None:
                    raise RuntimeError("Embedding failed")
                vectors.append(vec)
            return vectors, skipped
        if self._use is not None:
            return self._use(texts).numpy(), skipped
        assert self.vectorizer is not None
        return self.vectorizer.transform(texts), skipped

    def fit(self, dialogues: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> None:
        joined = [" ".join(d[-self.context_size :]) for d, _ in dialogues]
        labels = [set(l) for _, l in dialogues]
        if self.vectorizer is not None:
            texts, skipped = self._sanitize_texts(joined)
            if skipped:
                logger.info("skipping %d training inputs", len(skipped))
            labels = [lab for i, lab in enumerate(labels) if i not in skipped]
            if not texts:
                raise ValueError("No valid training data after sanitization")
            X = self.vectorizer.fit_transform(texts)
        else:
            X, skipped = self._embed(joined)
            if skipped:
                logger.info("skipping %d training inputs", len(skipped))
            labels = [lab for i, lab in enumerate(labels) if i not in skipped]
        Y = self.mlb.fit_transform(labels)
        self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.classifier.fit(X, Y)

    def predict(self, messages: Sequence[str]) -> Dict[str, float]:
        if not self.classifier:
            raise RuntimeError("Model not fitted")
        joined = " ".join(messages[-self.context_size :])
        if self.vectorizer is not None:
            texts, skipped = self._sanitize_texts([joined])
            if skipped:
                logger.warning("input skipped due to policy checks")
                return {}
            X = self.vectorizer.transform(texts)
        else:
            X, skipped = self._embed([joined])
            if skipped:
                logger.warning("input skipped due to policy checks")
                return {}
        probs = self.classifier.predict_proba(X)[0]
        labels = self.mlb.classes_
        scores = {label: float(p) * self.trend_bias[label] for label, p in zip(labels, probs)}
        return scores

    def classify(self, messages: Sequence[str]) -> List[Tuple[str, float]]:
        scores = self.predict(messages)
        intents = [(l, s) for l, s in scores.items() if s >= self.threshold]
        if not intents:
            intents.append(("clarify", 1.0))
        return sorted(intents, key=lambda x: x[1], reverse=True)

    def adjust_threshold(self, feedback_accuracy: float) -> None:
        if feedback_accuracy < 0.5:
            self.threshold = max(0.1, self.threshold * 0.9)
        else:
            self.threshold = min(0.9, self.threshold * 1.05)

    def update_trend_bias(self, counts: Dict[str, int]) -> None:
        total = sum(counts.values()) or 1
        for label, c in counts.items():
            self.trend_bias[label] = 1 + c / total

