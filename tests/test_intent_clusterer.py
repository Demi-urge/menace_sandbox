import sys
from pathlib import Path

import pytest

sys.modules['hdbscan'] = None

import intent_clusterer as ic  # noqa: E402
from vector_utils import cosine_similarity  # noqa: E402


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setattr(ic, "governed_embed", lambda text: [float(len(text))])


class DummyRetriever:
    def __init__(self):
        self.items = []

    def add_vector(self, vector, metadata):
        self.items.append({"vector": vector, "metadata": metadata})

    def search(self, vector, top_k=10):
        scored = [
            (item["metadata"], cosine_similarity(vector, item["vector"]))
            for item in self.items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [dict(path=m["path"], score=s) for m, s in scored[:top_k]]


def _write(tmp: Path, rel: str, content: str) -> None:
    p = tmp / rel
    p.write_text(content)


def test_index_and_cluster(tmp_path):
    _write(tmp_path, "a.py", '"""A"""\nimport b\n')
    _write(tmp_path, "b.py", '"""B"""\nimport a\n')
    _write(tmp_path, "c.py", '"""C"""\n')
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.index_repository(tmp_path)
    paths = {
        item["metadata"]["path"]: item["metadata"].get("cluster_id")
        for item in retr.items
    }
    assert paths["a.py"] == paths["b.py"]
    assert "cluster_id" in retr.items[0]["metadata"]


def test_cluster_search(tmp_path):
    _write(tmp_path, "a.py", '"""A doc"""\nimport b\n')
    _write(tmp_path, "b.py", '"""B doc"""\nimport a\n')
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(retr)
    clusterer.index_repository(tmp_path)
    cid = clusterer.cluster_map["a"]
    text, vec = clusterer.get_cluster_intents(cid)
    assert "A doc" in text and "B doc" in text
    res = clusterer.find_modules_related_to("cluster A", top_k=1)
    assert res and res[0]["cluster_id"] == cid


def test_collect_intent_no_docstring(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("# comment about func\n\ndef fn():\n    pass\n")
    clusterer = ic.IntentClusterer(DummyRetriever())
    text, meta = clusterer._collect_intent(path)
    assert meta["names"] == ["fn"]
    assert "docstrings" not in meta
    assert meta["comments"] == ["comment about func"]


def test_collect_intent_only_comments_module(tmp_path):
    path = tmp_path / "cmod.py"
    path.write_text("# just a comment\n")
    clusterer = ic.IntentClusterer(DummyRetriever())
    text, meta = clusterer._collect_intent(path)
    assert text == ""
    assert meta["names"] == []


def test_collect_intent_mixed_encoding(tmp_path):
    path = tmp_path / "latin.py"
    content = "# -*- coding: latin-1 -*-\n# espa\xf1ol\n\nclass X:\n    pass\n"
    path.write_bytes(content.encode("latin-1"))
    clusterer = ic.IntentClusterer(DummyRetriever())
    text, meta = clusterer._collect_intent(path)
    assert meta["names"] == ["X"]
    assert "espa\xf1ol" in meta["comments"][0]
