import types

import pytest

from vector_service.retriever import PatchRetriever
from vector_metrics_db import VectorMetricsDB


class DummyStore:
    def __init__(self, vectors, meta, metric="cosine"):
        self.vectors = vectors
        self.ids = [str(i) for i in range(len(vectors))]
        self.meta = meta
        self.metric = metric

    def query(self, vector, top_k=5):
        scores = []
        for vid, vec in zip(self.ids, self.vectors):
            if self.metric == "inner_product":
                score = sum(x * y for x, y in zip(vector, vec))
            else:  # cosine
                na = sum(x * x for x in vector) ** 0.5
                nb = sum(x * x for x in vec) ** 0.5
                if not na or not nb:
                    score = 0.0
                else:
                    score = sum(x * y for x, y in zip(vector, vec)) / (na * nb)
            scores.append((vid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _make_pr(metric):
    vectors = [[1.0, 0.0], [1000.0, 1.0], [0.0, 1.0]]
    meta = [
        {"origin_db": "patch", "metadata": {"text": "a"}},
        {"origin_db": "patch", "metadata": {"text": "b"}},
        {"origin_db": "patch", "metadata": {"text": "c"}},
    ]
    store = DummyStore(vectors, meta, metric=metric)

    def fake_vectorise(kind, record):
        return [1.0, 0.0]

    vec_service = types.SimpleNamespace(vectorise=fake_vectorise)
    return PatchRetriever(store=store, vector_service=vec_service, metric=metric)


def test_similarity_switching():
    pr = PatchRetriever(
        store=types.SimpleNamespace(),
        vector_service=types.SimpleNamespace(),
        metric="cosine",
    )
    assert pytest.approx(pr._similarity([1, 0], [1, 0])) == 1.0
    pr.metric = "inner_product"
    assert pr._similarity([2, 3], [4, 5]) == 2 * 4 + 3 * 5


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("cosine", ["0", "1", "2"]),
        ("inner_product", ["1", "0", "2"]),
    ],
)
def test_search_top_n_accuracy(metric, expected):
    pr = _make_pr(metric)
    results = pr.search("query", top_k=3)
    ids = [r["record_id"] for r in results]
    assert ids == expected
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_enhancement_score_boost():
    vectors = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    meta = [
        {"origin_db": "patch", "metadata": {"text": "a", "enhancement_score": 0.0}},
        {"origin_db": "patch", "metadata": {"text": "b", "enhancement_score": 1.0}},
        {"origin_db": "patch", "metadata": {"text": "c", "enhancement_score": 0.0}},
    ]
    store = DummyStore(vectors, meta)

    def fake_vectorise(kind, record):
        return [1.0, 0.0]

    vec_service = types.SimpleNamespace(vectorise=fake_vectorise)
    pr = PatchRetriever(
        store=store, vector_service=vec_service, enhancement_weight=1.0
    )
    results = pr.search("query", top_k=3)
    ids = [r["record_id"] for r in results]
    assert ids[0] == "1"
    assert results[0]["score"] > results[1]["score"]


def test_prioritizes_enhancement_score_on_tie():
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    meta = [
        {"origin_db": "patch", "metadata": {"text": "a", "enhancement_score": 0.0}},
        {"origin_db": "patch", "metadata": {"text": "b", "enhancement_score": 2.0}},
    ]
    store = DummyStore(vectors, meta)

    def fake_vectorise(kind, record):
        return [1.0, 0.0]

    vec_service = types.SimpleNamespace(vectorise=fake_vectorise)
    pr = PatchRetriever(
        store=store, vector_service=vec_service, enhancement_weight=1.0
    )
    results = pr.search("query", top_k=2)
    ids = [r["record_id"] for r in results]
    assert ids == ["1", "0"]


def test_uses_vector_metrics_db_for_enhancement_score(tmp_path):
    vectors = [[1.0, 0.0]]
    meta = [{"origin_db": "patch", "metadata": {"text": "a"}}]
    store = DummyStore(vectors, meta)

    def fake_vectorise(kind, record):
        return [1.0, 0.0]

    vec_service = types.SimpleNamespace(vectorise=fake_vectorise)
    pr = PatchRetriever(store=store, vector_service=vec_service, enhancement_weight=0.0)
    pr.vector_metrics = VectorMetricsDB(path=str(tmp_path / "vm.db"))
    pr.vector_metrics.record_patch_summary("0", enhancement_score=2.0)

    results = pr.search("query", top_k=1)
    assert results[0]["enhancement_score"] == pytest.approx(2.0)
