import types
import tempfile
from contextlib import contextmanager

from db_router import DBRouter
import universal_retriever


@contextmanager
def _retriever():
    with tempfile.TemporaryDirectory() as tmp:
        router = DBRouter("alpha", f"{tmp}/local.db", f"{tmp}/shared.db")
        conn = router.shared_conn
        conn.execute(
            "CREATE TABLE errors (id INTEGER PRIMARY KEY, source_menace_id TEXT, frequency REAL)"
        )
        conn.execute(
            "INSERT INTO errors (id, source_menace_id, frequency) VALUES (1, 'alpha', 7.0)"
        )
        error_db = types.SimpleNamespace(
            conn=conn, router=types.SimpleNamespace(menace_id="alpha")
        )
        ur = universal_retriever.UniversalRetriever.__new__(
            universal_retriever.UniversalRetriever
        )
        ur.error_db = error_db
        yield ur


def test_error_frequency_local():
    with _retriever() as ur:
        assert ur._error_frequency(1) == 7.0


def test_error_frequency_global():
    with _retriever() as ur:
        assert ur._error_frequency(1, scope="global") == 0.0


def test_error_frequency_all():
    with _retriever() as ur:
        assert ur._error_frequency(1, scope="all") == 7.0
