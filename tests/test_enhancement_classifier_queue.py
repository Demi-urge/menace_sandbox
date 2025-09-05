import contextlib
import sqlite3
import sys
import types


class _CtxConnWrapper:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @contextlib.contextmanager
    def _connect(self):
        yield self.conn


# Provide stubs so ``enhancement_classifier`` can import ``CodeDB`` and ``PatchHistoryDB``
sys.modules["code_database"] = types.SimpleNamespace(
    CodeDB=_CtxConnWrapper, PatchHistoryDB=_CtxConnWrapper
)

# Provide minimal vector_service stub so PatchSuggestionDB uses fallback mixin
sys.modules.setdefault("vector_service", types.SimpleNamespace(EmbeddableDBMixin=object))

from enhancement_classifier import EnhancementClassifier  # noqa: E402
from patch_suggestion_db import PatchSuggestionDB  # noqa: E402


def _build_dbs() -> tuple[_CtxConnWrapper, _CtxConnWrapper]:
    code_conn = sqlite3.connect(":memory:")  # noqa: SQL001
    code_conn.execute("CREATE TABLE code (id INTEGER PRIMARY KEY)")
    code_conn.executemany("INSERT INTO code (id) VALUES (?)", [(1,), (2,)])

    patch_conn = sqlite3.connect(":memory:")  # noqa: SQL001
    patch_conn.execute(
        """
        CREATE TABLE patch_history(
            code_id INTEGER,
            filename TEXT,
            roi_delta REAL,
            errors_before INTEGER,
            errors_after INTEGER,
            complexity_delta REAL
        )
        """
    )
    patch_conn.executemany(
        "INSERT INTO patch_history VALUES (1,'low.py',?,?,?,?)",  # path-ignore
        [(-5.0, 0, 0, 0.0), (-4.0, 0, 0, 0.0), (-6.0, 0, 0, 0.0)],
    )
    patch_conn.executemany(
        "INSERT INTO patch_history VALUES (2,'med.py',?,?,?,?)",  # path-ignore
        [(-1.0, 0, 0, 0.0), (-1.5, 0, 0, 0.0), (-0.5, 0, 0, 0.0)],
    )
    return _CtxConnWrapper(code_conn), _CtxConnWrapper(patch_conn)


def test_classifier_queue_and_top_suggestions(tmp_path) -> None:
    code_db, patch_db = _build_dbs()
    classifier = EnhancementClassifier(code_db=code_db, patch_db=patch_db)
    suggestions = list(classifier.scan_repo())
    assert len(suggestions) == 2
    by_path = {s.path: s for s in suggestions}
    assert by_path["low.py"].score > by_path["med.py"].score  # path-ignore

    db = PatchSuggestionDB(tmp_path / "s.db")
    db.queue_suggestions(suggestions)
    top = db.top_suggestions(2)
    assert [s.module for s in top] == ["low.py", "med.py"]  # path-ignore
    assert top[0].rationale == by_path["low.py"].rationale  # path-ignore
