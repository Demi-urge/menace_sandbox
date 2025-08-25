import importlib
import json
import sys

import pytest


def test_custom_categories_persist(monkeypatch, tmp_path):
    cat_file = tmp_path / "cats.json"
    cat_file.write_text(json.dumps(["support"]))
    monkeypatch.setenv("INTENT_CATEGORIES_FILE", str(cat_file))
    emb_file = tmp_path / "cat_embeddings.json"
    monkeypatch.setenv("INTENT_CATEGORY_EMBEDDINGS", str(emb_file))

    sys.modules.pop("intent_clusterer", None)
    import intent_clusterer as ic
    importlib.reload(ic)

    monkeypatch.setattr(ic, "governed_embed", lambda text, model=None: [1.0])
    assert ic._categorise(None, "support tool") == "support"
    data = json.loads(emb_file.read_text())
    assert "support" in data

    sys.modules.pop("intent_clusterer", None)
    import intent_clusterer as ic2
    importlib.reload(ic2)
    calls = {"count": 0}

    def _fake(text: str, model: str | None = None):
        calls["count"] += 1
        return [1.0]

    monkeypatch.setattr(ic2, "governed_embed", _fake)
    assert ic2._categorise(None, "support help") == "support"
    assert calls["count"] == 1
    sys.modules.pop("intent_clusterer", None)
