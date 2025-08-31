import sqlite3
import contextlib
import types
import sys


class _CtxConnWrapper:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _connect(self):
        @contextlib.contextmanager
        def cm():
            yield self.conn
        return cm()


class _PlainConnWrapper:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _connect(self):
        return self.conn


class DummyPatchSuggestionDB:
    def __init__(self) -> None:
        self.records = []

    def add(self, rec) -> None:
        self.records.append(rec)


class SuggestionRecord:
    def __init__(self, module: str, description: str) -> None:
        self.module = module
        self.description = description


class DummyAuditTrail:
    def __init__(self) -> None:
        self.events = []

    def record(self, event) -> None:
        self.events.append(event)


sys.modules['code_database'] = types.SimpleNamespace(
    CodeDB=_CtxConnWrapper, PatchHistoryDB=_CtxConnWrapper
)
sys.modules['chatgpt_enhancement_bot'] = types.SimpleNamespace(EnhancementDB=_PlainConnWrapper)
sys.modules['patch_suggestion_db'] = types.SimpleNamespace(
    SuggestionRecord=SuggestionRecord, PatchSuggestionDB=DummyPatchSuggestionDB
)
sys.modules['audit_trail'] = types.SimpleNamespace(AuditTrail=DummyAuditTrail)

from enhancement_classifier import EnhancementClassifier


def test_scan_repo_scores_and_queues_suggestions() -> None:
    code_conn = sqlite3.connect(":memory:")
    code_conn.execute("CREATE TABLE code (id INTEGER PRIMARY KEY)")
    code_conn.execute("INSERT INTO code (id) VALUES (1)")
    code_conn.execute(
        "CREATE TABLE code_enhancements (code_id INTEGER, enhancement_id INTEGER)"
    )
    code_conn.execute("INSERT INTO code_enhancements VALUES (1, 1)")

    patch_conn = sqlite3.connect(":memory:")
    patch_conn.execute(
        "CREATE TABLE patch_history (code_id INTEGER, filename TEXT, roi_delta REAL, complexity_delta REAL)"
    )
    patch_conn.executemany(
        "INSERT INTO patch_history VALUES (1, 'mod.py', ?, ?)",
        [(0.1, 0.2)] * 3,
    )

    enh_conn = sqlite3.connect(":memory:")
    enh_conn.execute(
        "CREATE TABLE enhancements (id INTEGER PRIMARY KEY, outcome_score REAL)"
    )
    enh_conn.execute("INSERT INTO enhancements VALUES (1, 0.5)")

    sugg_db = DummyPatchSuggestionDB()
    classifier = EnhancementClassifier(
        code_db=_CtxConnWrapper(code_conn),
        patch_db=_CtxConnWrapper(patch_conn),
        enhancement_db=_PlainConnWrapper(enh_conn),
        suggestion_db=sugg_db,
        audit_trail=DummyAuditTrail(),
    )

    suggestions = list(classifier.scan_repo())
    assert len(suggestions) == 1
    rec = sugg_db.records[0]
    assert rec.module == "mod.py"
    assert "score=2.60" in rec.description
    assert suggestions[0] is rec
