from vector_service.retriever import StackRetriever


class DummyBackend:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(self, embedding, k=0, similarity_threshold=0.0):  # pragma: no cover - simple stub
        self.calls.append(
            {
                "embedding": list(embedding),
                "k": k,
                "threshold": similarity_threshold,
            }
        )
        return [
            {
                "score": 0.8,
                "metadata": {
                    "language": "Python",
                    "summary": "py snippet",
                    "size": 80,
                    "redacted": True,
                },
            },
            {
                "score": 0.6,
                "metadata": {
                    "language": "JavaScript",
                    "summary": "js snippet",
                    "size": 40,
                    "redacted": True,
                },
            },
        ]


def test_stack_retriever_filters_languages():
    backend = DummyBackend()
    retriever = StackRetriever(backend=backend, top_k=5)

    hits = retriever.retrieve([0.1, 0.2], k=3, languages=["python"])

    assert len(hits) == 1
    assert hits[0]["metadata"]["language"] == "Python"
    assert backend.calls[0]["k"] == 3


def test_stack_retriever_limits_by_size():
    backend = DummyBackend()
    retriever = StackRetriever(backend=backend, top_k=5)

    hits = retriever.retrieve([0.1], max_lines=50)

    assert len(hits) == 1
    assert hits[0]["metadata"]["language"] == "JavaScript"
    assert hits[0]["metadata"]["size"] == 40
