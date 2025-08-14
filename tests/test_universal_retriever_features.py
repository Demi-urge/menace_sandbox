from types import MethodType

import pytest

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB
from menace.chatgpt_enhancement_bot import EnhancementDB, Enhancement
from menace.information_db import InformationDB, InformationRecord
from menace.universal_retriever import UniversalRetriever, boost_linked_candidates


def _fake_encoder(self, text: str):
    """Return deterministic vectors based on keywords for testing."""
    text = text.lower()
    if "bot" in text:
        return [1.0, 0.0]
    if "workflow" in text:
        return [0.8, 0.2]
    if "error" in text:
        return [0.6, 0.4]
    if "enhancement" in text:
        return [0.4, 0.6]
    if "info" in text:
        return [0.0, 1.0]
    return [0.5, 0.5]


def _constant_encoder(self, text: str):
    """Return a constant vector for tests not concerned with similarity."""
    return [1.0]


def _patch_encoders(*dbs):
    for db in dbs:
        db.encode_text = MethodType(_fake_encoder, db)


def test_cross_db_search(tmp_path):
    """All configured databases are queried and returned in results."""
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    enh_db = EnhancementDB(path=tmp_path / "enh.db", vector_index_path=tmp_path / "enh.idx")
    info_db = InformationDB(path=str(tmp_path / "info.db"), vector_index_path=str(tmp_path / "info.idx"))

    _patch_encoders(bot_db, wf_db, err_db, enh_db, info_db)

    bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))
    wf_db.add(WorkflowRecord(workflow=["workflow"], title="wf"))
    err_db.add_error("error")
    enh_db.add(Enhancement(idea="x", rationale="y", summary="enhancement", after_code="code"))
    info_db.add(InformationRecord(data_type="info", summary="info"))

    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=wf_db,
        error_db=err_db,
        enhancement_db=enh_db,
        information_db=info_db,
    )

    hits = retriever.retrieve("bot", top_k=5, link_multiplier=1.0)
    assert {h.origin_db for h in hits} == {"bot", "workflow", "error", "enhancement", "information"}


def test_metric_based_ranking(tmp_path):
    """Error frequency influences ranking when vector distance is tied."""
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    err_db.encode_text = MethodType(_constant_encoder, err_db)
    high_id = err_db.add_error("failure A")
    low_id = err_db.add_error("failure B")
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (10, high_id))
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (2, low_id))

    retriever = UniversalRetriever(error_db=err_db)
    hits = retriever.retrieve("anything", top_k=2)
    assert [h.record_id for h in hits] == [high_id, low_id]
    assert hits[0].reason.startswith("frequent error")
    assert hits[0].confidence > hits[1].confidence


def test_link_based_boosting(tmp_path):
    """Linked candidates receive a multiplier and linkage path."""
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")

    for db in (bot_db, wf_db, err_db):
        db.encode_text = MethodType(_constant_encoder, db)

    bot_id = bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))
    wf_id = wf_db.add(WorkflowRecord(workflow=["t"], title="wf", assigned_bots=[str(bot_id)]))
    bot_db.link_workflow(bot_id, wf_id)
    err_id = err_db.add_error("boom")
    err_db.conn.execute("INSERT INTO bot_error(bot_id, error_id) VALUES (?, ?)", (bot_id, err_id))

    scored = [
        {"source": "bot", "record_id": bot_id, "confidence": 1.0},
        {"source": "workflow", "record_id": wf_id, "confidence": 1.0},
        {"source": "error", "record_id": err_id, "confidence": 1.0},
        {"source": "information", "record_id": 99, "confidence": 1.0},
    ]
    paths = boost_linked_candidates(scored, bot_db=bot_db, error_db=err_db, multiplier=2.0)

    assert paths == {0: "bot->workflow->error", 1: "bot->workflow->error", 2: "bot->workflow->error"}
    for idx in (0, 1, 2):
        assert scored[idx]["confidence"] == pytest.approx(2.0)
    assert scored[3]["confidence"] == pytest.approx(1.0)
    assert 3 not in paths
