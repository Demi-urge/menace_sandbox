import pytest

from universal_retriever import UniversalRetriever


class DummyEncoder:
    def encode_text(self, text: str):
        return [0.0]


class DummyLookupDB:
    def get_vector(self, record_id):  # pragma: no cover - simple stub
        return None


class DummyRetriever(UniversalRetriever):
    def __init__(self, items):
        super().__init__(code_db=DummyEncoder(), enable_model_ranking=False, enable_reliability_bias=False)
        self._items = items

    def _retrieve_candidates(self, query, top_k, db_names=None):
        feats = {"distance": 0.0, "similarity": 0.9, "context": 0.0}
        return [("db", str(i), item, 0.0, feats) for i, item in enumerate(self._items, start=1)]


def test_retriever_skips_disallowed_license():
    items = [
        {"text": "This program is licensed under the GNU General Public License", "vector_id": "v1"},
        {"text": "regular text", "vector_id": "v2"},
    ]
    r = DummyRetriever(items)
    hits, _, _ = r.retrieve("q", top_k=2)
    assert len(hits) == 1
    assert hits[0].metadata.get("vector_id") == "v2"


def test_retriever_attaches_semantic_alerts():
    items = [
        {"text": "eval('data')", "vector_id": "v1"},
    ]
    r = DummyRetriever(items)
    hits, _, _ = r.retrieve("q", top_k=1)
    alerts = hits[0].metadata.get("semantic_alerts")
    assert alerts and any("eval" in a[1] for a in alerts)


def test_register_db_without_encode_text_does_not_replace_encoder():
    retriever = UniversalRetriever(
        code_db=DummyEncoder(),
        enable_model_ranking=False,
        enable_reliability_bias=False,
    )
    original_encoder = retriever._encoder
    retriever.register_db("lookup", DummyLookupDB(), ("id",))
    assert retriever._encoder is original_encoder
    assert "lookup" in retriever._non_encoder_dbs


def test_missing_encoder_raises_clear_error(monkeypatch):
    retriever = UniversalRetriever(
        bot_db=DummyLookupDB(),
        enable_model_ranking=False,
        enable_reliability_bias=False,
    )
    monkeypatch.setattr(
        "vector_service.embed_utils.get_text_embeddings", lambda *_args, **_kwargs: []
    )
    retriever._encoder = None
    with pytest.raises(RuntimeError, match="No text embedding backend available"):
        retriever._to_vector("query")


def test_text_embeddings_fallback_when_no_encoder(monkeypatch):
    retriever = UniversalRetriever(
        bot_db=DummyLookupDB(),
        enable_model_ranking=False,
        enable_reliability_bias=False,
    )
    monkeypatch.setattr(
        "vector_service.embed_utils.get_text_embeddings",
        lambda *_args, **_kwargs: [[0.25, 0.5]],
    )
    retriever._encoder = None
    assert retriever._to_vector("query") == [0.25, 0.5]
