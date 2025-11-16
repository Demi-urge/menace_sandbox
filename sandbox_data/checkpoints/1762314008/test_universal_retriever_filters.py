from universal_retriever import UniversalRetriever


class DummyEncoder:
    def encode_text(self, text: str):
        return [0.0]


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
