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
    monkeypatch.setattr(ndp, "get_embedder", lambda: _DummyEmbedder())
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

