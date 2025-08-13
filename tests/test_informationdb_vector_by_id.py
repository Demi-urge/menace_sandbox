from types import MethodType

from menace.information_db import InformationDB, InformationRecord


def test_vector_by_id(tmp_path):
    db = InformationDB(
        path=str(tmp_path / "info.db"),
        vector_backend="annoy",
        vector_index_path=str(tmp_path / "info.index"),
    )

    def fake_embed(self, text: str):
        return [float(len(text))]

    db._embed = MethodType(fake_embed, db)

    rec = InformationRecord(data_type="news", summary="alpha", keywords=["k1"])
    info_id = db.add(rec)
    assert db.vector(info_id) == db._metadata[str(info_id)]["vector"]

    # Remove stored vector to ensure fallback to DB fetch
    del db._metadata[str(info_id)]
    row = db.conn.execute(
        "SELECT * FROM information WHERE info_id=?", (info_id,)
    ).fetchone()
    expected = db._embed(db._embed_text(dict(row)))
    assert db.vector(info_id) == expected
