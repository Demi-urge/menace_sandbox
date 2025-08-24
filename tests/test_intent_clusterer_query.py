import intent_clusterer as ic


class DummyRetriever:
    def __init__(self):
        self.items = []

    def add_vector(self, vector, metadata):
        self.items.append({"vector": vector, "metadata": metadata})

    def search(self, vector, top_k=10):
        return [dict(metadata=item["metadata"], vector=item["vector"]) for item in self.items]


def test_query_normalizes_vectors(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = 1
    retr.add_vector([0.1, 0.0], {"path": "a.py", "cluster_id": 1})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 1.0])
    res = clusterer.query("anything", threshold=0.5)
    assert res and res[0].path == "a.py" and res[0].similarity > 0.6


def test_query_falls_back_to_clusters(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = 5
    retr.add_vector([0.0, 1.0], {"path": "a.py", "cluster_id": 5})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 0.0])
    monkeypatch.setattr(ic.IntentClusterer, "get_cluster_intents", lambda self, cid: ("", [1.0, 0.0]))
    res = clusterer.query("whatever", threshold=0.9)
    assert res and res[0].path is None and res[0].cluster_ids == [5]


def test_query_without_cluster_ids(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = 1
    retr.add_vector([1.0, 0.0], {"path": "a.py", "cluster_id": 1})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 0.0])
    res = clusterer.query("foo", include_clusters=False)
    assert res and res[0].cluster_ids == []
