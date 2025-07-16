import pytest
import menace.code_database as cdbm


def test_code_fts_search(tmp_path):
    db = cdbm.CodeDB(tmp_path / "c.db")
    if not getattr(db, "has_fts", False):
        pytest.skip("fts5 not available")
    rec1 = cdbm.CodeRecord(code="print('alpha')", summary="first alpha")
    db.add(rec1)
    rec2 = cdbm.CodeRecord(code="print('beta')", summary="second beta")
    db.add(rec2)
    res = db.search("alpha")
    summaries = [r["summary"] for r in res]
    assert "first alpha" in summaries

