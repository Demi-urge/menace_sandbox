import db_router
import pytest


def test_denied_tables_raise(tmp_path):
    router = db_router.DBRouter("deny", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    assert "capital_ledger" in db_router.DENY_TABLES
    with pytest.raises(ValueError):
        router.get_connection("capital_ledger")
    router.close()
