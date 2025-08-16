"""Tests for the :mod:`universal_retriever` helper."""

from types import MethodType
import pytest

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB
from menace.universal_retriever import ResultBundle, UniversalRetriever


def _keyword_encoder(self, text: str):
    """Return simple 2D embeddings based on keywords."""

    text = text.lower()
    if "bot" in text:
        return [1.0, 0.0]
    if "workflow" in text:
        return [0.8, 0.2]
    if "error" in text:
        return [0.6, 0.4]
    return [0.0, 1.0]


def _constant_encoder(self, text: str):
    """Return a constant vector so ranking relies purely on metrics."""

    return [1.0]


def test_cross_db_merge(tmp_path):
    """Results from different databases are merged into a single list."""

    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    for db in (bot_db, wf_db):
        db.encode_text = MethodType(_keyword_encoder, db)

    bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))
    wf_db.add(WorkflowRecord(workflow=["workflow"], title="alpha"))

    retriever = UniversalRetriever(bot_db=bot_db, workflow_db=wf_db)
    hits, session_id, vectors = retriever.retrieve("alpha", top_k=5, link_multiplier=1.0)

    assert session_id and vectors
    assert {h.origin_db for h in hits} == {"bot", "workflow"}
    # feature logging: ensure each hit includes similarity and contextual metrics
    for h in hits:
        assert "similarity" in h.metadata
        assert "contextual_metrics" in h.metadata


def test_metric_weighting_prioritises_frequent_errors(tmp_path):
    """Errors with higher frequency outrank lower frequency ones."""

    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    err_db.encode_text = MethodType(_constant_encoder, err_db)

    high_id = err_db.add_error("boom")
    low_id = err_db.add_error("bust")
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (10, high_id))
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (1, low_id))

    retriever = UniversalRetriever(error_db=err_db)
    hits, _, _ = retriever.retrieve("anything", top_k=2)

    assert [h.record_id for h in hits] == [high_id, low_id]
    assert hits[0].reason.startswith("frequent error recurrence")
    assert hits[0].score > hits[1].score
    metrics = hits[0].metadata["contextual_metrics"]
    assert metrics["model_score"] == pytest.approx(1.0)
    assert metrics["frequency"] == pytest.approx(10.0)


def test_relationship_boosting(tmp_path):
    """Linked results receive a confidence boost and linkage path."""

    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")

    for db in (bot_db, wf_db, err_db):
        db.encode_text = MethodType(_keyword_encoder, db)

    bot_id = bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))
    wf_id = wf_db.add(WorkflowRecord(workflow=["workflow"], assigned_bots=[str(bot_id)], performance_data='{"runs": 5}'))
    bot_db.conn.execute("INSERT INTO bot_workflow(bot_id, workflow_id) VALUES (?, ?)", (bot_id, wf_id))

    err_id = err_db.add_error("error")
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (3, err_id))
    err_db.conn.execute("INSERT INTO bot_error(bot_id, error_id) VALUES (?, ?)", (bot_id, err_id))

    bot_db.conn.commit()
    err_db.conn.commit()

    retriever = UniversalRetriever(bot_db=bot_db, workflow_db=wf_db, error_db=err_db)

    baseline, _, _ = retriever.retrieve("bot", top_k=3, link_multiplier=1.0)
    boosted, _, _ = retriever.retrieve("bot", top_k=3, link_multiplier=1.5)

    assert all(isinstance(h, ResultBundle) for h in boosted)
    assert {h.origin_db for h in boosted} == {"bot", "workflow", "error"}

    base = {h.origin_db: h.score for h in baseline}
    boost = {h.origin_db: h.score for h in boosted}
    for src in ("bot", "workflow", "error"):
        assert boost[src] > base[src]
        assert "contextual_metrics" in next(h.metadata for h in boosted if h.origin_db == src)

    reasons = {h.origin_db: h.reason for h in boosted}
    path = "linked via bot->workflow->error"
    assert path in reasons["workflow"]
    assert path in reasons["error"]

    links = {h.origin_db: h.metadata.get("linked_records", []) for h in boosted}
    assert len(links["bot"]) == 2
    assert len(links["workflow"]) == 2
    assert len(links["error"]) == 2
