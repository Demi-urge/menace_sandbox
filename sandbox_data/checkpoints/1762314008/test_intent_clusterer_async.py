from pathlib import Path

import pytest
import sys
import types

st_stub = types.ModuleType("sentence_transformers")
st_stub.SentenceTransformer = None
sys.modules.setdefault("sentence_transformers", st_stub)

import intent_clusterer as ic


@pytest.fixture(autouse=True)
def mock_summariser(monkeypatch):
    """Replace the heavy summariser with a deterministic stub."""

    monkeypatch.setattr(
        ic, "summarise_texts", lambda texts, **_: "cluster helper summary"
    )


class DummyRetriever:
    def __init__(self):
        self.items = []

    def register_db(self, *args, **kwargs):
        pass

    def add_vector(self, vector, metadata):
        self.items.append({"vector": vector, "metadata": metadata})

    def search(self, vector, top_k=10):
        return [dict(metadata=item["metadata"], vector=item["vector"]) for item in self.items]


@pytest.mark.asyncio
async def test_query_async_normalizes_vectors(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [1]
    retr.add_vector([0.1, 0.0], {"path": "a.py", "cluster_ids": [1]})  # path-ignore
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 1.0])
    res = await clusterer.query_async("anything", threshold=0.5)
    assert res and res[0].path == "a.py" and res[0].similarity > 0.6  # path-ignore


@pytest.mark.asyncio
async def test_find_modules_related_to_async(monkeypatch):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.cluster_map["a"] = [1]
    retr.add_vector([0.1, 0.0], {"path": "a.py", "cluster_ids": [1]})  # path-ignore
    monkeypatch.setattr(ic, "governed_embed", lambda text: [1.0, 1.0])
    res = await clusterer.find_modules_related_to_async("anything", top_k=1)
    assert res and res[0].path == "a.py"  # path-ignore


@pytest.mark.asyncio
async def test_find_clusters_related_to_async_from_existing_store(monkeypatch, tmp_path: Path):
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    member = tmp_path / "m.py"  # path-ignore
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
    res = await fresh.find_clusters_related_to_async("cluster helper", top_k=1)
    assert res and res[0].cluster_ids == [7] and res[0].origin == "cluster"
