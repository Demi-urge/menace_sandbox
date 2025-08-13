from types import MethodType

import pytest

from menace.error_bot import ErrorDB
from menace.error_logger import TelemetryEvent


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = ErrorDB(
        path=tmp_path / "e.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"e.{backend}.index",
    )
    db.metadata_path = tmp_path / "e.meta.json"
    db._metadata.clear()
    db._id_map.clear()
    db._index = None

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    ev1 = TelemetryEvent(root_cause="alpha", stack_trace="trace1")
    db.add_telemetry(ev1)
    rec_id = db.conn.execute("SELECT id FROM telemetry WHERE cause='alpha'").fetchone()[0]
    assert str(rec_id) in db._metadata
    assert captured and "alpha" in captured[0] and "trace1" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0]["cause"] == "alpha"


    # insert telemetry without embedding and backfill
    db.conn.execute(
        "INSERT INTO telemetry(cause, stack_trace) VALUES ('beta', 'trace2')"
    )
    db.conn.commit()
    new_id = db.conn.execute("SELECT id FROM telemetry WHERE cause='beta'").fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0]["cause"] == "beta"

    err_id = db.add_error("gamma")
    assert db._metadata[str(err_id)]["source_id"] == str(err_id)
