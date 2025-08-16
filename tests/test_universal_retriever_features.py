from types import MethodType
import os
import sys
import types

import pytest

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB

# ``chatgpt_enhancement_bot`` pulls in ``run_autonomous`` which performs
# system dependency checks during import.  Stub that module to avoid requiring
# ffmpeg/qemu/tesseract in the test environment.
sys.modules.setdefault(
    "menace.run_autonomous", types.SimpleNamespace(_verify_required_dependencies=lambda: None, LOCAL_KNOWLEDGE_MODULE=None)
)

from menace.chatgpt_enhancement_bot import EnhancementDB, Enhancement
from menace.information_db import InformationDB, InformationRecord
from menace.universal_retriever import UniversalRetriever, boost_linked_candidates
from menace.deployment_bot import DeploymentDB


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

    hits, session_id, vectors = retriever.retrieve("bot", top_k=5, link_multiplier=1.0)
    assert session_id and vectors
    assert {h.origin_db for h in hits} == {"bot", "workflow", "error", "enhancement", "information"}
    # every hit should now expose similarity and contextual metrics for feature
    # inspection and dataset generation
    for h in hits:
        assert "similarity" in h.metadata
        cm = h.metadata.get("contextual_metrics", {})
        assert cm.get("model_score", 0.0) > 0.0


def test_metric_based_ranking(tmp_path):
    """Error frequency influences ranking when vector distance is tied."""
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    err_db.encode_text = MethodType(_constant_encoder, err_db)
    high_id = err_db.add_error("failure A")
    low_id = err_db.add_error("failure B")
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (10, high_id))
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (2, low_id))

    retriever = UniversalRetriever(error_db=err_db)
    hits, _, _ = retriever.retrieve("anything", top_k=2)
    assert [h.record_id for h in hits] == [high_id, low_id]
    assert hits[0].reason.startswith("frequent error recurrence")
    assert hits[0].confidence > hits[1].confidence
    metrics = hits[0].metadata["contextual_metrics"]
    assert metrics["frequency"] == pytest.approx(10.0)
    assert metrics["model_score"] == pytest.approx(1.0)


def test_enhancement_roi_metric(tmp_path):
    """Enhancement ROI influences ranking when similarity ties."""
    enh_db = EnhancementDB(path=tmp_path / "e.db", vector_index_path=tmp_path / "e.idx")
    enh_db.encode_text = MethodType(_constant_encoder, enh_db)
    high = Enhancement(idea="x", rationale="y", summary="s", score=10.0, cost_estimate=2.0)
    low = Enhancement(idea="x", rationale="y", summary="s", score=4.0, cost_estimate=3.0)
    high_id = enh_db.add(high)
    low_id = enh_db.add(low)

    retriever = UniversalRetriever(enhancement_db=enh_db)
    hits, _, _ = retriever.retrieve("anything", top_k=2)
    assert [h.record_id for h in hits] == [str(high_id), str(low_id)]
    assert hits[0].reason.startswith("high ROI uplift")
    assert hits[0].confidence > hits[1].confidence
    metrics = hits[0].metadata["contextual_metrics"]
    assert metrics["roi"] == pytest.approx(8.0)


def test_workflow_usage_metric(tmp_path):
    """Workflow usage counts affect ranking."""
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    wf_db.encode_text = MethodType(_constant_encoder, wf_db)
    high_id = wf_db.add(WorkflowRecord(workflow=["a"], title="x", assigned_bots=["1","2","3"]))
    low_id = wf_db.add(WorkflowRecord(workflow=["a"], title="y", assigned_bots=["1"]))

    retriever = UniversalRetriever(workflow_db=wf_db)
    hits, _, _ = retriever.retrieve("anything", top_k=2)
    assert [h.record_id for h in hits] == [high_id, low_id]
    assert hits[0].reason.startswith("heavy usage")
    assert hits[0].confidence > hits[1].confidence
    metrics = hits[0].metadata["contextual_metrics"]
    assert metrics["usage"] == pytest.approx(3.0)


def test_bot_deploy_frequency_metric(tmp_path):
    """Bots with more deployments rank higher."""
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    bot_db.encode_text = MethodType(_constant_encoder, bot_db)
    high_id = bot_db.add_bot(BotRecord(name="a", purpose="b"))
    low_id = bot_db.add_bot(BotRecord(name="c", purpose="d"))

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        dep_db = DeploymentDB(path="deployment.db")
        for _ in range(3):
            dep_db.conn.execute(
                "INSERT INTO bot_trials(bot_id, deploy_id, status, ts) VALUES (?,?,?,?)",
                (high_id, 1, "ok", "now"),
            )
        dep_db.conn.execute(
            "INSERT INTO bot_trials(bot_id, deploy_id, status, ts) VALUES (?,?,?,?)",
            (low_id, 1, "ok", "now"),
        )
        dep_db.conn.commit()

        retriever = UniversalRetriever(bot_db=bot_db)
        hits, _, _ = retriever.retrieve("anything", top_k=2)
    finally:
        os.chdir(cwd)

    assert [h.record_id for h in hits] == [high_id, low_id]
    assert hits[0].reason.startswith("widely deployed bot")
    assert hits[0].confidence > hits[1].confidence
    metrics = hits[0].metadata["contextual_metrics"]
    assert metrics["deploy"] == pytest.approx(3.0)


def test_link_based_boosting(tmp_path):
    """Linked candidates receive a multiplier and linkage path."""
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    info_db = InformationDB(path=tmp_path / "info.db", vector_index_path=tmp_path / "info.idx")

    for db in (bot_db, wf_db, err_db, info_db):
        db.encode_text = MethodType(_constant_encoder, db)

    bot_id = bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))
    wf_id = wf_db.add(WorkflowRecord(workflow=["t"], title="wf", assigned_bots=[str(bot_id)]))
    bot_db.link_workflow(bot_id, wf_id)
    err_id = err_db.add_error("boom")
    err_db.conn.execute("INSERT INTO bot_error(bot_id, error_id) VALUES (?, ?)", (bot_id, err_id))

    info_db.conn.execute(
        "CREATE TABLE information_bots(info_id INTEGER, bot_id INTEGER)"
    )
    info_id = info_db.add(InformationRecord(data_type="info", summary="i"))
    info_db.conn.execute(
        "INSERT INTO information_bots(info_id, bot_id) VALUES (?, ?)",
        (info_id, bot_id),
    )

    scored = [
        {"source": "bot", "record_id": bot_id, "confidence": 1.0},
        {"source": "workflow", "record_id": wf_id, "confidence": 1.0},
        {"source": "error", "record_id": err_id, "confidence": 1.0},
        {"source": "information", "record_id": info_id, "confidence": 1.0},
    ]
    paths = boost_linked_candidates(
        scored,
        bot_db=bot_db,
        error_db=err_db,
        information_db=info_db,
        multiplier=2.0,
    )

    for idx in (0, 1, 2, 3):
        path, link_ids = paths[idx]
        assert path == "bot->workflow->error->information"
        assert len(link_ids) == 3
        assert scored[idx]["confidence"] == pytest.approx(2.0)
