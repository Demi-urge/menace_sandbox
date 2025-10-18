import menace.nlp_domain_parser as ndp


class _DummyEmbedder:
    def encode(self, texts):
        vecs = []
        for text in texts:
            vec = [0.0] * len(ndp.TARGET_ANCHORS)
            for i, kw in enumerate(ndp.TARGET_ANCHORS):
                if kw.split()[0] in text:
                    vec[i] = 1.0
                    break
            vecs.append(vec)
        return vecs


def _reset(monkeypatch):
    monkeypatch.setattr(ndp, "get_embedder", lambda timeout=None: _DummyEmbedder())
    monkeypatch.setattr(
        ndp,
        "governed_embed",
        lambda text, embedder=None: embedder.encode([text])[0],
    )
    class _DummyNP:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        class linalg:  # type: ignore[valid-type]
            @staticmethod
            def norm(values):
                return sum(x * x for x in values) ** 0.5

    monkeypatch.setattr(ndp, "np", _DummyNP())
    ndp._MODEL = ndp._VECTORIZER = ndp._ANCHOR_VECS = ndp._METHOD = None


def test_classify_basic(monkeypatch):
    _reset(monkeypatch)
    ndp.load_model()
    res = ndp.classify_text("military operations")
    assert res
    assert res[0][0] == "military"
    assert ndp.flag_if_similar("military base", threshold=0.5)


def test_classify_entry(monkeypatch):
    _reset(monkeypatch)
    entry = {"target_domain": "military", "action_description": "strategy"}
    res = ndp.classify_text(entry)
    assert res
    assert res[0][0] == "military"


def test_load_model_uses_timeout_and_falls_back(monkeypatch):
    calls: list[float | None] = []

    def fake_get_embedder(timeout=None):
        calls.append(timeout)
        return None

    class _DummyVectorizer:
        def fit(self, anchors):
            return self

        def transform(self, anchors):
            return anchors

    def _fake_similarity(vec, anchors):
        return [[0.0 for _ in anchors]]

    monkeypatch.setattr(ndp, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(ndp, "TfidfVectorizer", lambda: _DummyVectorizer())
    monkeypatch.setattr(ndp, "cosine_similarity", _fake_similarity)
    ndp._MODEL = ndp._VECTORIZER = ndp._ANCHOR_VECS = ndp._METHOD = None

    method = ndp.load_model()

    assert calls and calls[0] == ndp._EMBEDDER_TIMEOUT
    assert method == "tfidf"

