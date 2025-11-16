from menace_sandbox import intent_vectorizer as iv


def test_intent_vectorizer_init_does_not_fetch_embedder(monkeypatch):
    calls: list[str] = []

    def fake_get_embedder():
        calls.append("called")
        return None

    monkeypatch.setattr(iv, "get_embedder", fake_get_embedder)

    vectorizer = iv.IntentVectorizer()

    assert calls == []
    assert vectorizer._embedder is None
    assert vectorizer._embedder_loaded is False


def test_intent_vectorizer_encode_uses_shared_embedder(monkeypatch):
    vectorizer = iv.IntentVectorizer()

    get_calls: list[str] = []
    shared_embedder = object()

    def fake_get_embedder():
        get_calls.append("get")
        return shared_embedder

    embed_calls: list[tuple[str, object]] = []

    def fake_governed_embed(text: str, embedder: object):
        embed_calls.append((text, embedder))
        return [1.0, 2.0]

    monkeypatch.setattr(iv, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(iv, "governed_embed", fake_governed_embed)

    result = vectorizer._encode("hello world")
    assert result == [1.0, 2.0]

    result_again = vectorizer._encode("hello world")
    assert result_again == [1.0, 2.0]

    assert get_calls == ["get"]
    assert embed_calls == [("hello world", shared_embedder), ("hello world", shared_embedder)]


def test_intent_vectorizer_encode_falls_back_to_local(monkeypatch):
    vectorizer = iv.IntentVectorizer()

    get_calls: list[str] = []

    def fake_get_embedder():
        get_calls.append("get")
        return None

    def fail_governed_embed(_text: str, _embedder: object):
        raise AssertionError("governed_embed should not be called when embedder missing")

    local_calls: list[str] = []

    def fake_local_embed(text: str):
        local_calls.append(text)
        return [9.0, 8.0]

    monkeypatch.setattr(iv, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(iv, "governed_embed", fail_governed_embed)
    monkeypatch.setattr(iv, "_local_embed", fake_local_embed)

    result = vectorizer._encode("fallback text")
    assert result == [9.0, 8.0]

    assert get_calls == ["get"]
    assert local_calls == ["fallback text"]
    assert vectorizer._embedder is None
    assert vectorizer._embedder_loaded is True
