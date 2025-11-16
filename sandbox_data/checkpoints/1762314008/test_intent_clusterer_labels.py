import intent_clusterer as ic


def test_rake_produces_informative_label():
    texts = [
        "Authentication helper handles user login",
        "Provides login helper for authentication",
    ]
    rake_label = ic.summarise_texts(texts, method="rake", top_k=1)
    tfidf_label = ic.summarise_texts(texts, method="tfidf", top_k=1)

    assert rake_label == "authentication helper handles user login"
    assert tfidf_label == "authentication"

    label, summary = ic.derive_cluster_label(texts, top_k=1)
    assert (label, summary) == ("authentication helper handles user login",) * 2


def test_derive_cluster_label_falls_back(monkeypatch):
    def fake_summarise(texts, method="tfidf", top_k=5):
        if method == "rake":
            raise RuntimeError("rake unavailable")
        return "tfidf summary"

    monkeypatch.setattr(ic, "summarise_texts", fake_summarise)

    label, summary = ic.derive_cluster_label(["auth module"], top_k=1)
    assert (label, summary) == ("tfidf summary",) * 2

