import logging

from menace_sandbox import research_storage as rs


def test_infodb_migrations_apply_once_per_epoch(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(rs._INFO_DB_MIGRATION_EPOCH_ENV, "epoch-latch")
    rs._INFO_DB_MIGRATION_LATCH.clear()

    db_path = tmp_path / "info.db"
    with caplog.at_level(logging.DEBUG, logger=rs.__name__):
        db1 = rs.InfoDB(path=db_path, vector_index_path=tmp_path / "i1.idx")
        db2 = rs.InfoDB(path=db_path, vector_index_path=tmp_path / "i2.idx")

    assert db1._schema_initialised is True
    assert db2._schema_initialised is True
    assert sum("InfoDB migrations completed" in r.message for r in caplog.records) == 1
    assert any("migration application skipped" in r.message for r in caplog.records)
