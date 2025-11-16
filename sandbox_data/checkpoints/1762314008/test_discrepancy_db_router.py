import sys
import types

import pytest

# Stub vector_service to avoid heavy model loading


class DummyMixin:
    def __init__(self, *a, **k):
        pass

    def encode_text(self, text):
        return [0.0]

    def add_embedding(self, *a, **k):
        pass

    def save_index(self):
        pass

    def backfill_embeddings(self, *a, **k):
        pass

    def search_by_vector(self, *a, **k):
        return []


sys.modules.setdefault("vector_service", types.SimpleNamespace(EmbeddableDBMixin=DummyMixin))

from discrepancy_db import DiscrepancyDB, DiscrepancyRecord  # noqa: E402
from db_router import DBRouter, DENY_TABLES  # noqa: E402


def test_discrepancy_scope_utils(tmp_path):
    shared_db = tmp_path / "shared.db"
    router1 = DBRouter("one", str(tmp_path / "one.db"), str(shared_db))
    router2 = DBRouter("two", str(tmp_path / "two.db"), str(shared_db))
    try:
        db1 = DiscrepancyDB(router=router1, vector_index_path=tmp_path / "i1.index")
        db2 = DiscrepancyDB(router=router2, vector_index_path=tmp_path / "i2.index")
        rec = DiscrepancyRecord(message="oops")
        rid = db1.add(rec)
        fetched = db2.get(rid, scope="all")
        assert fetched and fetched.message == "oops"
        assert db2.get(rid, scope="global")
        assert db2.get(rid, scope="local") is None
    finally:
        router1.close()
        router2.close()


def test_discrepancy_db_respects_denied_table(tmp_path):
    shared_db = tmp_path / "shared.db"
    router = DBRouter("one", str(tmp_path / "local.db"), str(shared_db))
    DENY_TABLES.add("discrepancies")
    try:
        with pytest.raises(ValueError):
            DiscrepancyDB(router=router, vector_index_path=tmp_path / "d.index")
    finally:
        DENY_TABLES.discard("discrepancies")
        router.close()
