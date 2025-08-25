import intent_clusterer as ic
import json
import pickle
import sqlite3
import pytest


class DummyRetriever:
    def __init__(self):
        self.items = []

    def register_db(self, *args, **kwargs):
        pass

    def add_vector(self, vector, metadata):
        self.items.append({"vector": vector, "metadata": metadata})

    def search(self, vector, top_k=10):
        return [dict(metadata=item["metadata"], vector=item["vector"]) for item in self.items]


def test_query_normalizes_vectors(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [1]
    retr.add_vector([0.1, 0.0], {"path": "a.py", "cluster_ids": [1]})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 1.0])
    res = clusterer.query("anything", threshold=0.5)
    assert res and res[0].path == "a.py" and res[0].similarity > 0.6


def test_query_falls_back_to_clusters(monkeypatch, tmp_path):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [5]
    retr.add_vector([0.0, 1.0, 0.0], {"path": "a.py", "cluster_ids": [5]})
    monkeypatch.setattr(
        ic,
        "governed_embed",
        lambda text: [
            float("auth" in text.lower()),
            float("pay" in text.lower()),
            float("help" in text.lower()),
        ],
    )
    # Persist a cluster record with a matching vector so fallback can occur
    member = tmp_path / "b.py"
    member.write_text('"""cluster helper"""')
    blob = sqlite3.Binary(pickle.dumps([1.0, 0.0, 0.0]))
    meta = {
        "members": [str(member)],
        "kind": "cluster",
        "cluster_ids": [5],
        "path": "cluster:5",
        "text": "cluster helper",
        "label": "auth helper",
        "summary": "",
    }
    clusterer.conn.execute(
        "REPLACE INTO intent_embeddings (module_path, vector, metadata) VALUES (?, ?, ?)",
        ("cluster:5", blob, json.dumps(meta)),
    )
    clusterer.conn.commit()
    res = clusterer.query("auth", threshold=0.9)
    assert res and res[0].path is None and res[0].cluster_ids == [5] and res[0].category in ic.CANONICAL_CATEGORIES
    text, vec = clusterer.get_cluster_intents(5)
    assert "cluster helper" in text
    assert vec == [1.0, 0.0, 0.0]


def test_query_without_cluster_ids(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [1]
    retr.add_vector([1.0, 0.0], {"path": "a.py", "cluster_ids": [1]})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 0.0])
    res = clusterer.query("foo", include_clusters=False)
    assert res and res[0].cluster_ids == []


@pytest.fixture
def multi_cluster_clusterer(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [1, 2]
    retr.add_vector([1.0, 0.0], {"path": "a.py", "cluster_ids": [1, 2]})
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 0.0])
    return clusterer


def test_query_returns_multiple_cluster_ids(multi_cluster_clusterer):
    res = multi_cluster_clusterer.query("whatever", threshold=0.2)
    assert res and res[0].cluster_ids == [1, 2]


def test_find_clusters_related_to_from_existing_store(monkeypatch, tmp_path):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    member = tmp_path / "m.py"
    member.write_text('"""cluster helper"""')
    monkeypatch.setattr(
        ic,
        "governed_embed",
        lambda text: [
            float("auth" in text.lower()),
            float("pay" in text.lower()),
            float("help" in text.lower()),
        ],
    )
    clusterer.vectors[str(member)] = [1.0, 0.0, 0.0]
    clusterer._index_clusters({"7": [str(member)]})
    fresh = ic.IntentClusterer(retr)
    res = fresh.find_clusters_related_to("cluster helper", top_k=1)
    assert res and res[0].cluster_ids == [7] and res[0].category in ic.CANONICAL_CATEGORIES
