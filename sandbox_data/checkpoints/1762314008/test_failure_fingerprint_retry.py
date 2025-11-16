import json
import sqlite3
import types
from pathlib import Path
import pytest

from . import test_self_coding_engine_chunking as setup  # reuse stub environment

sce = setup.sce


SAMPLE_TRACE = (
    'Traceback (most recent call last):\n'
    '  File "mod.py", line 1, in <module>\n'  # path-ignore
    '    1/0\n'
    'ZeroDivisionError: division by zero'
)


class DummyPatchDB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE patch_history (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "filename TEXT, description TEXT, outcome TEXT, tests_passed INTEGER)"
        )
        self.router = types.SimpleNamespace(get_connection=lambda name: self.conn)

    def record_vector_metrics(self, *a, **k):
        return None


class DummyFP:
    def __init__(self, filename, function_name, stack_trace, error_message, prompt_text, timestamp=0.0):
        self.filename = filename
        self.function_name = function_name
        self.stack_trace = stack_trace
        self.error_message = error_message
        self.prompt_text = prompt_text
        self.embedding = [0.1]
        self.timestamp = timestamp

    @classmethod
    def from_failure(cls, filename, function_name, stack_trace, error_message, prompt_text):
        return cls(filename, function_name, stack_trace, error_message, prompt_text)


def _build_engine(monkeypatch, tmp_path, similar_return, skip: bool = False):
    patch_db = DummyPatchDB(tmp_path / "ph.db")
    engine = sce.SelfCodingEngine(
        code_db=object(),
        memory_mgr=object(),
        patch_db=patch_db,
        skip_retry_on_similarity=skip,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    records: list[dict] = []
    engine.audit_trail = types.SimpleNamespace(record=lambda payload: records.append(payload))
    engine._build_retry_context = lambda desc, rep: {}
    calls: list[str] = []

    def fake_apply(self, path, description, context_meta=None, **kwargs):
        calls.append(description)
        if len(calls) == 1:
            self._last_retry_trace = SAMPLE_TRACE
            return None, False, 0.0
        return 1, False, 0.0

    engine.apply_patch = types.MethodType(fake_apply, engine)
    monkeypatch.setattr(sce, "FailureFingerprint", DummyFP)
    monkeypatch.setattr(sce, "log_fingerprint", lambda fp: None)

    def fake_check(fp, store, thresh, desc):
        skip_flag = skip or len(similar_return) >= 3
        return desc, skip_flag, 0.9, similar_return, ""

    monkeypatch.setattr(sce, "check_similarity_and_warn", fake_check)
    return engine, records, calls, patch_db


def test_retry_prompt_adjusted_for_similar_failure(monkeypatch, tmp_path):
    prior = DummyFP("prev.py", "main", "t", "Boom", "p", timestamp=1.0)  # path-ignore
    eng, records, calls, db = _build_engine(monkeypatch, tmp_path, [prior])

    pid, reverted, delta = eng.apply_patch_with_retry(Path("f.py"), "desc", max_attempts=2)  # path-ignore
    assert pid == 1 and not reverted
    assert any("WARNING" in d for d in calls[1:])
    assert any("retry_adjusted" in r for r in records)
    row = db.conn.execute("SELECT outcome FROM patch_history").fetchall()
    assert ("retry_adjusted",) in row


def test_retry_skipped_after_similarity_limit(monkeypatch, tmp_path):
    prev = [DummyFP("a.py", "f", "t", "Boom", "p", timestamp=i + 1) for i in range(3)]  # path-ignore
    eng, records, calls, db = _build_engine(monkeypatch, tmp_path, prev)

    pid, reverted, delta = eng.apply_patch_with_retry(Path("f.py"), "desc", max_attempts=3)  # path-ignore
    assert pid is None and len(calls) == 1
    assert any(r.get("reason") == "retry_skipped_due_to_similarity" for r in records)
    row = db.conn.execute("SELECT description, outcome FROM patch_history").fetchall()
    assert row and row[0][1] == "retry_skipped"
    info = json.loads(row[0][0])
    assert info["reason"] == "retry_skipped_due_to_similarity"
    assert "fingerprint_hash" in info and "similarity" in info and "cluster_id" in info


def test_retry_skipped_when_configured(monkeypatch, tmp_path):
    prior = DummyFP("prev.py", "main", "t", "Boom", "p", timestamp=1.0)  # path-ignore
    eng, records, calls, db = _build_engine(monkeypatch, tmp_path, [prior], skip=True)

    pid, reverted, delta = eng.apply_patch_with_retry(Path("f.py"), "desc", max_attempts=2)  # path-ignore
    assert pid is None and len(calls) == 1
    assert any(r.get("reason") == "retry_skipped_due_to_similarity" for r in records)
    row = db.conn.execute("SELECT description, outcome FROM patch_history").fetchall()
    assert row and row[0][1] == "retry_skipped"
    info = json.loads(row[0][0])
    assert info["reason"] == "retry_skipped_due_to_similarity"
    assert "fingerprint_hash" in info and "similarity" in info and "cluster_id" in info
