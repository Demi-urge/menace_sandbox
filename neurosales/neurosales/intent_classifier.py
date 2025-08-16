from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
from collections import defaultdict

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

    def _embed(self, texts: Sequence[str]):
        if self._sbert is not None:
            return self._sbert.encode(texts)
        if self._use is not None:
            return self._use(texts).numpy()
        assert self.vectorizer is not None
        return self.vectorizer.transform(texts)

    def fit(self, dialogues: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> None:
        joined = [" ".join(d[-self.context_size :]) for d, _ in dialogues]
        labels = [set(l) for _, l in dialogues]
        if self.vectorizer is not None:
            X = self.vectorizer.fit_transform(joined)
        else:
            X = self._embed(joined)
        Y = self.mlb.fit_transform(labels)
        self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.classifier.fit(X, Y)

    def predict(self, messages: Sequence[str]) -> Dict[str, float]:
        if not self.classifier:
            raise RuntimeError("Model not fitted")
        joined = " ".join(messages[-self.context_size :])
        if self.vectorizer is not None:
            X = self.vectorizer.transform([joined])
        else:
            X = self._embed([joined])
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

