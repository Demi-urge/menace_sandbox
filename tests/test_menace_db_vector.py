import json
import pytest

pytest.importorskip("sqlalchemy")

from menace.databases import MenaceDB


def test_query_vector(tmp_path):
    db = MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with db.engine.begin() as conn:
        conn.execute(
            db.memory_embeddings.insert().values(
                id=1,
                key="k1",
                data="hello",
                version=1,
                tags="",
                ts="t",
                embedding=json.dumps([1.0, 0.0]),
            )
        )
        conn.execute(
            db.memory_embeddings.insert().values(
                id=2,
                key="k2",
                data="world",
                version=1,
                tags="",
                ts="t",
                embedding=json.dumps([0.0, 1.0]),
            )
        )
    res = db.query_vector([1.0, 0.0], limit=1)
    assert res and res[0]["key"] == "k1"

