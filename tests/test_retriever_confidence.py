from types import MethodType
import hashlib

from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.error_bot import ErrorDB
from menace.error_logger import TelemetryEvent
from menace.chatgpt_enhancement_bot import EnhancementDB, Enhancement, EnhancementHistory
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

    enh_db = EnhancementDB(path=tmp_path / "enh.db", vector_index_path=tmp_path / "enh.index")
    enh_db.encode_text = MethodType(fake_encode, enh_db)
    enh = Enhancement(idea="i", rationale="r", summary="bot", score=0.2, timestamp="t", context="", after_code="x")
    enh_id = enh_db.add(enh)
    enh_db.record_history(
        EnhancementHistory(
            file_path="f",
            original_hash="o",
            enhanced_hash=hashlib.sha1(b"x").hexdigest(),
            metric_delta=0.5,
            author_bot="b",
        )
    )
    bot_db.link_enhancement(bot_id, enh_id)

    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=wf_db,
        error_db=err_db,
        enhancement_db=enh_db,
    )

    res = retriever.retrieve_with_confidence("bot", top_k=5)
    assert res
    assert all("confidence" in r for r in res)
    for r in res:
        assert 0.0 <= r["confidence"] <= 1.0
