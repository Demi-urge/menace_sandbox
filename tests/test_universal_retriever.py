from types import MethodType

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB
from menace.chatgpt_enhancement_bot import EnhancementDB, Enhancement
from menace.information_db import InformationDB, InformationRecord
from menace.universal_retriever import UniversalRetriever, RetrievedItem


def test_universal_retriever_ranking_and_boosting(tmp_path):
    # create lightweight in-memory databases
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    err_db = ErrorDB(path=tmp_path / "err.db", vector_index_path=tmp_path / "err.idx")
    enh_db = EnhancementDB(path=tmp_path / "enh.db", vector_index_path=tmp_path / "enh.idx")
    info_db = InformationDB(path=str(tmp_path / "info.db"), vector_index_path=str(tmp_path / "info.idx"))

    def fake_encode(self, text: str):
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

    for db in (bot_db, wf_db, err_db, enh_db, info_db):
        db.encode_text = MethodType(fake_encode, db)

    # seed bot record
    bot_id = bot_db.add_bot(BotRecord(name="alpha", purpose="bot"))

    # workflow linked to bot
    wf_rec = WorkflowRecord(workflow=["workflow"], assigned_bots=[str(bot_id)], performance_data='{"runs": 10}')
    wf_id = wf_db.add(wf_rec)
    bot_db.conn.execute("INSERT INTO bot_workflow(bot_id, workflow_id) VALUES (?, ?)", (bot_id, wf_id))

    # error linked to bot with frequency metric
    err_id = err_db.add_error("error")
    err_db.conn.execute("UPDATE errors SET frequency=? WHERE id=?", (5, err_id))
    err_db.conn.execute("INSERT INTO bot_error(bot_id, error_id) VALUES (?, ?)", (bot_id, err_id))

    # enhancement linked to bot with ROI metric
    enh = Enhancement(idea="x", rationale="y", summary="enhancement", after_code="code", score=0.8)
    enh_id = enh_db.add(enh)
    bot_db.conn.execute("INSERT INTO bot_enhancement(bot_id, enhancement_id) VALUES (?, ?)", (bot_id, enh_id))

    # information record (unlinked)
    info_id = info_db.add(InformationRecord(data_type="info", summary="info"))

    bot_db.conn.commit()
    err_db.conn.commit()

    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=wf_db,
        error_db=err_db,
        enhancement_db=enh_db,
        information_db=info_db,
    )

    baseline = retriever.retrieve("bot", top_k=5, link_multiplier=1.0)
    boosted = retriever.retrieve("bot", top_k=5, link_multiplier=1.2)

    # structured result bundles
    assert all(isinstance(h, RetrievedItem) for h in boosted)

    # ranking reflects metrics and boosting
    sources = [h.origin_db for h in boosted]
    assert sources == ["bot", "workflow", "error", "enhancement", "information"]

    reasons = {h.origin_db: h.reason for h in boosted}
    path = "linked via bot->workflow->enhancement->error"
    assert reasons["error"].startswith("frequent error")
    assert path in reasons["error"]
    assert reasons["enhancement"].startswith("high ROI uplift")
    assert path in reasons["enhancement"]
    assert reasons["workflow"].startswith("heavy usage")
    assert path in reasons["workflow"]
    assert reasons["bot"].startswith("widely deployed bot")
    assert path in reasons["bot"]
    assert reasons["information"] == "relevant match"

    # metric weighting keeps information result least confident
    info_conf = next(h.confidence for h in boosted if h.origin_db == "information")
    assert all(h.confidence > info_conf for h in boosted if h.origin_db != "information")

    # chain boosting increases confidence for linked entries
    base_conf = {h.origin_db: h.confidence for h in baseline}
    boost_conf = {h.origin_db: h.confidence for h in boosted}
    for src in ("bot", "workflow", "error", "enhancement"):
        assert boost_conf[src] > base_conf[src]
    assert boost_conf["information"] == base_conf["information"]
