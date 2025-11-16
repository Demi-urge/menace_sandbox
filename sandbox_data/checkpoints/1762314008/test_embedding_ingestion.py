import json
from vector_utils import persist_embedding


def test_persist_embedding_writes_json(tmp_path):
    path = tmp_path / "emb.jsonl"
    persist_embedding("bot", "b1", [0.1, 0.2], path=path)
    data = json.loads(path.read_text().strip())
    assert data["type"] == "bot"
    assert data["id"] == "b1"
    assert data["vector"] == [0.1, 0.2]
