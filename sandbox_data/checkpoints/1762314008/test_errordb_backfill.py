from types import MethodType

from menace.error_bot import ErrorDB


def test_backfill_embeddings(tmp_path):
    db = ErrorDB(
        path=tmp_path / "e.db",
        vector_backend="annoy",
        vector_index_path=tmp_path / "e.index",
    )

    # deterministic embedding function
    def fake_embed(self, text: str):
        if "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    # insert records directly without embeddings
    db.conn.execute("INSERT INTO errors(id, message) VALUES (1, 'alpha error')")
    db.conn.execute(
        "INSERT INTO telemetry(id, cause, stack_trace, bot_id) VALUES (2, 'beta', 'trace', 'b1')"
    )
    db.conn.commit()

    db.backfill_embeddings(batch_size=10)

    # ensure embeddings were created for both tables
    res_err = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res_err and res_err[0].get("message") == "alpha error"

    res_tel = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res_tel and res_tel[0].get("cause") == "beta"
