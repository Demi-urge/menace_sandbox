import pytest
from pathlib import Path

import json
import sys
import types

st_stub = types.ModuleType("sentence_transformers")
st_stub.SentenceTransformer = None
sys.modules.setdefault("sentence_transformers", st_stub)

import menace_sandbox.embeddable_db_mixin as edm
import intent_db
import intent_clusterer as ic
from db_router import init_db_router, LOCAL_TABLES


def _fake(text: str, model=None) -> list[float]:
    lower = text.lower()
    return [float(lower.count("alpha")), float(lower.count("beta"))]


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(edm, "governed_embed", _fake)
    monkeypatch.setattr(ic, "governed_embed", _fake)
    monkeypatch.setattr(
        ic, "summarise_texts", lambda texts, **_: "alpha beta summary"
    )


def test_synergy_cluster_embeddings_and_query(tmp_path: Path, monkeypatch):
    (tmp_path / "a.py").write_text('"""alpha"""')  # path-ignore
    (tmp_path / "b.py").write_text('"""beta"""')  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"a": 1, "b": 1}))

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def __init__(self):
            self.items = []

        def register_db(self, *args, **kwargs):
            pass

        def add_vector(self, vector, metadata):
            self.items.append({"vector": list(vector), "metadata": dict(metadata)})

        def search(self, vector, top_k=10):
            return self.items

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_repository(tmp_path)

    res = clusterer.find_clusters_related_to("alpha beta", top_k=5)
    assert res and res[0].origin == "cluster"
    members = {str(tmp_path / "a.py"), str(tmp_path / "b.py")}  # path-ignore
    assert set(res[0].members or []) == members
    assert res[0].intent_text
    row = clusterer.conn.execute(
        "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
        (res[0].path,),
    ).fetchone()
    meta = json.loads(row[0])
    assert meta.get("intent_text")
    assert set(meta.get("members", [])) == members


def test_query_falls_back_to_cluster_vectors(tmp_path: Path):
    (tmp_path / "a.py").write_text('"""alpha"""')  # path-ignore
    (tmp_path / "b.py").write_text('"""beta"""')  # path-ignore

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def __init__(self):
            self.items = []

        def register_db(self, *args, **kwargs):
            pass

        def add_vector(self, vector, metadata):
            if metadata.get("kind") == "cluster":
                return
            path = metadata.get("path")
            for item in self.items:
                if item["metadata"].get("path") == path:
                    item["metadata"].update(metadata)
                    item["vector"] = list(vector)
                    return
            self.items.append({"vector": list(vector), "metadata": dict(metadata)})

        def search(self, vector, top_k=10):
            return self.items[:1]

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_modules([tmp_path / "a.py", tmp_path / "b.py"])  # path-ignore
    clusterer.cluster_intents(1)

    matches = clusterer.query("beta", threshold=0.6)
    assert matches and matches[0].path is None
    assert matches[0].cluster_ids == [0]
    text, vec = clusterer.get_cluster_intents(0)
    assert "alpha" in text and "beta" in text
    assert vec == [0.5, 0.5]
