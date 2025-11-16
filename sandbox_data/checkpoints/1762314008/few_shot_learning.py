from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

try:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional deps
    np = None
    TfidfVectorizer = None  # type: ignore


class FewShotZeroShotClassifier:
    """Simple few-shot and zero-shot text classifier using embeddings."""

    def __init__(self, *, threshold: float = 0.5, micro_epoch: int = 5) -> None:
        self.threshold = threshold
        self.micro_epoch = micro_epoch
        self.categories: Dict[str, List[str]] = {}
        self.centroids: Dict[str, List[float]] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self._pending: List[Tuple[str, str]] = []
        self.misclass: Dict[str, int] = {}

    # ----------------------- internal helpers -----------------------
    def _ensure_vectorizer(self, texts: Iterable[str]) -> None:
        if TfidfVectorizer is None:
            return
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.vectorizer.fit(list(texts))
        else:
            self.vectorizer.fit(list(texts))

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if np is None or TfidfVectorizer is None:
            # fallback: simple bag-of-words counts
            vocab: Dict[str, int] = {}
            embeds: List[List[float]] = []
            for text in texts:
                counts: Dict[str, int] = {}
                for tok in text.lower().split():
                    counts[tok] = counts.get(tok, 0) + 1
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
                vec = [0.0] * len(vocab)
                for tok, c in counts.items():
                    idx = vocab[tok]
                    vec[idx] = float(c)
                embeds.append(vec)
            return embeds
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.vectorizer.fit(texts)
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()

    def _recompute_centroids(self) -> None:
        all_texts: List[str] = []
        for texts in self.categories.values():
            all_texts.extend(texts)
        if not all_texts:
            return
        self._ensure_vectorizer(all_texts)
        idx = 0
        embeddings = self._embed(all_texts)
        self.centroids = {}
        for label, texts in self.categories.items():
            count = len(texts)
            sub = embeddings[idx : idx + count]
            idx += count
            if np is not None:
                centroid = np.array(sub).mean(axis=0)
                self.centroids[label] = centroid.tolist()
            else:
                # simple python average
                if not sub:
                    continue
                dim = len(sub[0])
                sums = [0.0] * dim
                for vec in sub:
                    for i in range(dim):
                        sums[i] += vec[i]
                self.centroids[label] = [s / len(sub) for s in sums]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if np is not None:
            a1, b1 = np.array(a), np.array(b)
            denom = np.linalg.norm(a1) * np.linalg.norm(b1) + 1e-8
            if denom == 0:
                return 0.0
            return float(a1.dot(b1) / denom)
        # fallback pure python
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        denom = na * nb + 1e-8
        if denom == 0:
            return 0.0
        return dot / denom

    # ----------------------- public API -----------------------
    def add_examples(self, label: str, examples: Iterable[str]) -> None:
        self.categories.setdefault(label, []).extend(examples)
        self._recompute_centroids()

    def classify(self, text: str) -> Tuple[str, float]:
        if not self.centroids:
            new_label = f"category_{len(self.categories)+1}"
            self.add_examples(new_label, [text])
            return new_label, 0.0
        query_vec = self._embed([text])[0]
        best_label = ""
        best_sim = -1.0
        for label, centroid in self.centroids.items():
            sim = self._cosine(query_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        if best_sim < self.threshold:
            new_label = f"category_{len(self.categories)+1}"
            self.add_examples(new_label, [text])
            self.misclass[new_label] = 0
            return new_label, best_sim
        return best_label, best_sim

    def log_feedback(self, text: str, predicted: str, correct: str) -> None:
        if correct not in self.categories:
            self.categories[correct] = []
        if predicted != correct:
            if text in self.categories.get(predicted, []):
                self.categories[predicted].remove(text)
                if not self.categories[predicted]:
                    self.categories.pop(predicted)
                    self.centroids.pop(predicted, None)
                    self.misclass.pop(predicted, None)
            self.categories[correct].append(text)
        else:
            self.categories[correct].append(text)
        self.misclass[predicted] = self.misclass.get(predicted, 0) + 1
        if self.misclass[predicted] >= self.micro_epoch:
            self._recompute_centroids()
            self.misclass[predicted] = 0


