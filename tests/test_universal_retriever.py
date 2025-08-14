from types import MethodType

from menace.bot_database import BotDB, BotRecord
from menace.information_db import InformationDB, InformationRecord
from menace.universal_retriever import UniversalRetriever


def test_universal_retriever(tmp_path):
    bot_db = BotDB(
        path=tmp_path / "b.db",
        vector_index_path=tmp_path / "b.index",
    )
    info_db = InformationDB(
        path=str(tmp_path / "i.db"),
        vector_index_path=str(tmp_path / "i.index"),
    )

    def fake_encode(self, text: str):
        if "bot" in text:
            return [1.0, 0.0]
        if "info" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    bot_db.encode_text = MethodType(fake_encode, bot_db)
    info_db.encode_text = MethodType(fake_encode, info_db)

    bot_rec = BotRecord(name="botty", purpose="bot")
    bot_db.add_bot(bot_rec)

    info_rec = InformationRecord(data_type="info", summary="some info")
    info_id = info_db.add(info_rec)

    retriever = UniversalRetriever(bot_db=bot_db, information_db=info_db)

    res_text = retriever.retrieve("bot", top_k=5)
    assert any(r.source_db == "bot" and r.metadata.get("name") == "botty" for r in res_text)

    res_obj = retriever.retrieve({"info_id": info_id}, top_k=5)
    assert res_obj and res_obj[0].record_id == info_id
