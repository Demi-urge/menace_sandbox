from types import MethodType
import pytest

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB
from menace.error_logger import TelemetryEvent
from menace.universal_retriever import UniversalRetriever


def test_retrieve_with_confidence(tmp_path):
    def fake_encode(self, text: str):
        return [1.0, 0.0] if "bot" in text else [0.0, 1.0]

    bot_db = BotDB(path=tmp_path / "b.db", vector_index_path=tmp_path / "b.index")
    bot_db.encode_text = MethodType(fake_encode, bot_db)
    bot_id = bot_db.add_bot(BotRecord(name="bot", purpose="bot"))

    wf_db = WorkflowDB(path=tmp_path / "w.db", vector_index_path=tmp_path / "w.index")
    wf_db.encode_text = MethodType(fake_encode, wf_db)
    wf = WorkflowRecord(
        workflow=["bot"],
        performance_data="{\"runs\": 3}",
        assigned_bots=[str(bot_id)],
        title="bot workflow",
    )
    wf_id = wf_db.add(wf)
    bot_db.link_workflow(bot_id, wf_id)

    err_db = ErrorDB(tmp_path / "e.db", vector_index_path=tmp_path / "e.index")
    err_db.encode_text = MethodType(fake_encode, err_db)
    err_db.add_telemetry(TelemetryEvent(root_cause="bot", stack_trace="trace"))

    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=wf_db,
        error_db=err_db,
    )

    res = retriever.retrieve_with_confidence("bot", top_k=5)
    assert res
    assert all("confidence" in r for r in res)
    for r in res:
        assert r["confidence"] >= 0.0


def test_linked_boost(tmp_path):
    def fake_encode(self, text: str):
        return [1.0]

    bot_db = BotDB(path=tmp_path / "b.db", vector_index_path=tmp_path / "b.index")
    bot_db.encode_text = MethodType(fake_encode, bot_db)
    bot_id = bot_db.add_bot(BotRecord(name="bot", purpose="bot"))

    wf_db = WorkflowDB(path=tmp_path / "w.db", vector_index_path=tmp_path / "w.index")
    wf_db.encode_text = MethodType(fake_encode, wf_db)
    wf = WorkflowRecord(workflow=["t"], assigned_bots=[str(bot_id)], title="wf")
    wf_id = wf_db.add(wf)
    bot_db.link_workflow(bot_id, wf_id)

    err_db = ErrorDB(path=tmp_path / "e.db", vector_index_path=tmp_path / "e.index")
    err_db.encode_text = MethodType(fake_encode, err_db)
    err_id = err_db.add_error("err", bots=[str(bot_id)])

    retriever = UniversalRetriever(
        bot_db=bot_db, workflow_db=wf_db, error_db=err_db
    )

    base = retriever.retrieve_with_confidence("q", top_k=5, link_multiplier=1.0)
    boosted = retriever.retrieve_with_confidence("q", top_k=5, link_multiplier=2.0)

    def _find(res, kind):
        for r in res:
            item = r["item"]
            if kind == "bot" and isinstance(item, dict) and item.get("id") == bot_id:
                return r["confidence"]
            if kind == "workflow" and hasattr(item, "wid") and item.wid == wf_id:
                return r["confidence"]
            if kind == "error" and isinstance(item, dict) and item.get("id") == err_id:
                return r["confidence"]
        return None

    for k in ("bot", "workflow", "error"):
        c_base = _find(base, k)
        c_boost = _find(boosted, k)
        assert c_base is not None and c_boost is not None
        assert c_boost == pytest.approx(c_base * 2.0)
