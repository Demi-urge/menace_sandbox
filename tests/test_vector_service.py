import pytest
import importlib
import types
import json
from vector_service import (
    Retriever,
    PatchLogger,
    EmbeddingBackfill,
    EmbeddableDBMixin,
    RateLimitError,
    MalformedPromptError,
    FallbackResult,
)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DummyHit:
    def __init__(self, score=0.9, metadata=None):
        self.origin_db = "db"
        self.record_id = "1"
        self.score = score
        base_meta = {"redacted": True}
        if metadata:
            base_meta.update(metadata)
        self.metadata = base_meta


class DummyRetrieverSuccess:
    def retrieve_with_confidence(self, query, top_k):
        return [DummyHit()], 0.9, None


class DummyRetrieverRateLimit:
    def retrieve(self, query, top_k):
        raise Exception("429 rate limit")


class DummyRetrieverNoHits:
    def retrieve_with_confidence(self, query, top_k):
        return [], 0.0, None


class DummyRetrieverLowConfidence:
    def retrieve_with_confidence(self, query, top_k):
        return [DummyHit(score=0.05)], 0.05, None


def test_retriever_search_success_returns_results():
    r = Retriever(retriever=DummyRetrieverSuccess())
    results = r.search("query", session_id="s")
    assert results[0]["record_id"] == "1"


def test_retriever_search_rate_limit_raises_error():
    r = Retriever(retriever=DummyRetrieverRateLimit())
    with pytest.raises(RateLimitError):
        r.search("q", session_id="s")


def test_retriever_search_fallback_when_no_hits():
    r = Retriever(retriever=DummyRetrieverNoHits())
    res = r.search("q", session_id="s")
    assert isinstance(res, FallbackResult)
    assert res.reason == "no results"


def test_retriever_search_low_confidence_fallback():
    r = Retriever(retriever=DummyRetrieverLowConfidence())
    res = r.search("q", session_id="s")
    assert isinstance(res, FallbackResult)
    assert res.reason == "low confidence"


def test_retriever_search_malformed_prompt():
    r = Retriever(retriever=DummyRetrieverSuccess())
    with pytest.raises(MalformedPromptError):
        r.search("  ", session_id="s")


# ---------------------------------------------------------------------------
# PatchLogger
# ---------------------------------------------------------------------------


class DummyMetricsDB:
    def __init__(self, fail=False):
        self.fail = fail
        self.called = False
        self.args = None

    def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
        if self.fail:
            raise Exception("db failure")
        self.called = True
        self.args = (patch_id, result, pairs, session_id)


class DummyVectorMetricsDB:
    def __init__(self):
        self.called = False

    def update_outcome(
        self,
        session_id,
        pairs,
        *,
        contribution=0.0,
        patch_id="",
        win=False,
        regret=False,
    ):
        self.called = True

    def log_retrieval_feedback(self, db, *, win=False, regret=False, roi=0.0):
        pass


class DummyPatchDB:
    def __init__(self):
        self.called = False
        self.kwargs = None

    def record_vector_metrics(
        self,
        session_id,
        pairs,
        *,
        patch_id,
        contribution,
        roi_delta=None,
        win,
        regret,
        lines_changed=None,
        tests_passed=None,
        context_tokens=None,
        patch_difficulty=None,
        effort_estimate=None,
        enhancement_name=None,
        start_time=None,
        time_to_completion=None,
        timestamp=None,
        roi_deltas=None,
        errors=None,
        error_trace_count=None,
        roi_tag=None,
        enhancement_score=None,
        diff=None,
        summary=None,
        outcome=None,
    ):
        self.called = True
        self.kwargs = {
            "session_id": session_id,
            "pairs": pairs,
            "patch_id": patch_id,
            "contribution": contribution,
            "roi_delta": roi_delta,
            "win": win,
            "regret": regret,
            "lines_changed": lines_changed,
            "tests_passed": tests_passed,
            "context_tokens": context_tokens,
            "patch_difficulty": patch_difficulty,
            "effort_estimate": effort_estimate,
            "enhancement_name": enhancement_name,
            "start_time": start_time,
            "time_to_completion": time_to_completion,
            "timestamp": timestamp,
            "roi_deltas": roi_deltas,
            "errors": errors,
            "error_trace_count": error_trace_count,
            "roi_tag": roi_tag,
            "enhancement_score": enhancement_score,
            "diff": diff,
            "summary": summary,
            "outcome": outcome,
        }


def test_patch_logger_metrics_db_success():
    db = DummyMetricsDB()
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b"], True, patch_id="p", session_id="s")
    assert db.called
    assert db.args[2] == [("a", "b")]


def test_patch_logger_vector_metrics_db_success():
    vm = DummyVectorMetricsDB()
    pl = PatchLogger(vector_metrics=vm)
    pl.track_contributors(["x"], False, session_id="s")
    assert vm.called


def test_patch_logger_patch_db_success():
    pdb = DummyPatchDB()
    pl = PatchLogger(patch_db=pdb)
    pl.track_contributors(["a:b"], True, patch_id="7", session_id="sid")
    assert pdb.called
    assert pdb.kwargs["patch_id"] == 7
    assert pdb.kwargs["context_tokens"] == 0
    assert pdb.kwargs["patch_difficulty"] == 0


def test_patch_logger_metrics_db_failure_ignored():
    db = DummyMetricsDB(fail=True)
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b"], True, patch_id="p", session_id="s")


# ---------------------------------------------------------------------------
# EmbeddingBackfill
# ---------------------------------------------------------------------------


class DummyDB:
    def __init__(self, vector_backend="annoy"):
        self.batch = None

    def backfill_embeddings(self, batch_size=0):
        self.batch = batch_size


class ErrorDB(DummyDB):
    def backfill_embeddings(self, batch_size=0):
        raise RuntimeError("boom")


class InitFallbackDB:
    instances = []

    def __init__(self, vector_backend=""):
        if vector_backend:
            raise TypeError("no backend")
        self.__class__.instances.append(self)
        self.processed = False

    def backfill_embeddings(self, batch_size=0):
        self.processed = True
        self.batch = batch_size


def test_embedding_backfill_run_processes_databases(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [DummyDB]
    )
    called = []

    def fake_process(self, db, *, batch_size, session_id=""):
        called.append((db.__class__, batch_size, session_id))

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill(batch_size=7)
    eb.run(session_id="sid")
    assert called == [(DummyDB, 7, "sid")]


def test_embedding_backfill_run_continues_on_error(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [ErrorDB, DummyDB]
    )
    order = []

    def fake_process(self, db, *, batch_size, session_id=""):
        order.append(db.__class__)
        if isinstance(db, ErrorDB):
            raise RuntimeError("fail")

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill()
    eb.run(session_id="s")
    assert order == [ErrorDB, DummyDB]


def test_embedding_backfill_instantiation_fallback(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [InitFallbackDB]
    )
    eb = EmbeddingBackfill(batch_size=3)
    eb.run()
    assert InitFallbackDB.instances and InitFallbackDB.instances[0].processed
    assert InitFallbackDB.instances[0].batch == 3


def test_embedding_backfill_filters_by_db(monkeypatch):
    class WorkflowDB(DummyDB):
        pass

    class OtherDB(DummyDB):
        pass

    def fake_load(self, names=None):
        classes = [WorkflowDB, OtherDB]
        if names:
            keys = [n.lower().rstrip("s") for n in names]
            filtered = []
            for cls in classes:
                name = cls.__name__.lower()
                base = name[:-2] if name.endswith("db") else name
                for key in keys:
                    if name.startswith(key) or base.startswith(key):
                        filtered.append(cls)
                        break
            return filtered
        return classes

    monkeypatch.setattr(EmbeddingBackfill, "_load_known_dbs", fake_load)
    called = []

    def fake_process(self, db, *, batch_size, session_id=""):
        called.append(db.__class__)

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill()
    eb.run(db="workflows")
    assert called == [WorkflowDB]


def test_embedding_backfill_run_with_dbs(monkeypatch, tmp_path):
    class InformationDB(EmbeddableDBMixin):
        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            pass

    class CodeDB(EmbeddableDBMixin):
        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            pass

    class DiscrepancyDB(EmbeddableDBMixin):
        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            pass

    modules = {
        "information_db": types.ModuleType("information_db"),
        "code_database": types.ModuleType("code_database"),
        "discrepancy_db": types.ModuleType("discrepancy_db"),
    }
    modules["information_db"].InformationDB = InformationDB
    modules["code_database"].CodeDB = CodeDB
    modules["discrepancy_db"].DiscrepancyDB = DiscrepancyDB

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name in modules:
            return modules[name]
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    reg = tmp_path / "registry.json"
    reg.write_text(
        json.dumps(
            {
                "information": {"module": "information_db", "class": "InformationDB"},
                "code": {"module": "code_database", "class": "CodeDB"},
                "discrepancy": {"module": "discrepancy_db", "class": "DiscrepancyDB"},
            }
        )
    )
    import vector_service.embedding_backfill as eb_mod
    monkeypatch.setattr(eb_mod, "_REGISTRY_FILE", reg)
    monkeypatch.setattr(eb_mod.pkgutil, "walk_packages", lambda *a, **k: [])

    called = []

    def fake_process(self, db, *, batch_size, session_id=""):
        called.append(db.__class__)

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)

    eb = EmbeddingBackfill()
    subclasses = eb._load_known_dbs(names=["information", "code", "discrepancy"])
    assert subclasses == [InformationDB, CodeDB, DiscrepancyDB]

    eb.run(dbs=["information", "code", "discrepancy"])
    assert called == [InformationDB, CodeDB, DiscrepancyDB]


def test_watch_event_bus_triggers_backfill(monkeypatch):
    from vector_service.embedding_backfill import watch_event_bus, EmbeddingBackfill
    import time

    calls: list[tuple[list[str] | None, int | None, str]] = []

    def fake_run(self, *, dbs=None, batch_size=None, backend=None, session_id="", trigger="manual"):
        calls.append((dbs, batch_size, trigger))

    monkeypatch.setattr(EmbeddingBackfill, "run", fake_run)

    class DummyBus:
        def __init__(self) -> None:
            self._subs = []

        def subscribe(self, _topic, cb):
            self._subs.append(cb)

        def publish(self, topic, event):
            for cb in list(self._subs):
                cb(topic, event)

    bus = DummyBus()
    with watch_event_bus(bus=bus, batch_size=1):
        bus.publish("db:record_added", {"db": "info"})
        time.sleep(0.1)

    assert calls and calls[0][0] == ["info"]
    assert calls[0][1] == 1
    assert calls[0][2] == "event"
