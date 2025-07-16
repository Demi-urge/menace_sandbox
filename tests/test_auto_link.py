import sqlite3
import types
import sys
import logging
import pytest

pytest.importorskip("jinja2")

mods = {
    "menace.chatgpt_enhancement_bot": ["EnhancementDB", "ChatGPTEnhancementBot", "Enhancement"],
    "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
    "menace.text_research_bot": ["TextResearchBot"],
    "menace.video_research_bot": ["VideoResearchBot"],
    "menace.chatgpt_research_bot": ["ChatGPTResearchBot", "Exchange", "summarise_text"],
    "menace.database_manager": ["get_connection", "DB_PATH"],
    "menace.capital_management_bot": ["CapitalManagementBot"],
}
for name, attrs in mods.items():
    module = types.ModuleType(name)
    for attr in attrs:
        if attr == "summarise_text":
            setattr(module, attr, lambda text, ratio=0.2: text[:10])
        elif attr == "get_connection":
            setattr(module, attr, lambda path: None)
        elif attr == "DB_PATH":
            setattr(module, attr, ":memory:")
        else:
            setattr(module, attr, type(attr, (), {}))
    sys.modules.setdefault(name, module)

import menace.bot_database as bdbm
import menace.code_database as cdbm
import menace.research_aggregator_bot as rab
import menace.error_bot as eb
from menace.auto_link import auto_link


def test_auto_link(tmp_path):
    bot_db = bdbm.BotDB(tmp_path / "bots.db")
    code_db = cdbm.CodeDB(tmp_path / "code.db")
    info_db = rab.InfoDB(tmp_path / "info.db")
    err_db = eb.ErrorDB(tmp_path / "err.db")

    bid = bot_db.add_bot(
        bdbm.BotRecord(name="b"), models=[1], workflows=[2], enhancements=[3]
    )
    with sqlite3.connect(tmp_path / "bots.db") as conn:
        assert conn.execute("SELECT model_id FROM bot_model WHERE bot_id=?", (bid,)).fetchone()[0] == 1
        assert conn.execute("SELECT workflow_id FROM bot_workflow WHERE bot_id=?", (bid,)).fetchone()[0] == 2
        assert conn.execute("SELECT enhancement_id FROM bot_enhancement WHERE bot_id=?", (bid,)).fetchone()[0] == 3

    cid = code_db.add(
        cdbm.CodeRecord(code="x", summary="s"), bots=[bid], enhancements=[6], errors=[5]
    )
    with sqlite3.connect(code_db.path) as conn:
        assert conn.execute("SELECT bot_id FROM code_bots WHERE code_id=?", (cid,)).fetchone()[0] == bid
        assert conn.execute("SELECT enhancement_id FROM code_enhancements WHERE code_id=?", (cid,)).fetchone()[0] == 6
        assert conn.execute("SELECT error_id FROM code_errors WHERE code_id=?", (cid,)).fetchone()[0] == 5

    info_id = info_db.add(
        rab.ResearchItem(topic="t", content="c", timestamp=0.0),
        workflows=[7],
        enhancements=[8],
    )
    with sqlite3.connect(info_db.path) as conn:
        assert conn.execute("SELECT workflow_id FROM info_workflows WHERE info_id=?", (info_id,)).fetchone()[0] == 7
        assert conn.execute("SELECT enhancement_id FROM info_enhancements WHERE info_id=?", (info_id,)).fetchone()[0] == 8

    err_id = err_db.add_error("boom", models=[9], bots=[bid], codes=[cid])
    with sqlite3.connect(tmp_path / "err.db") as conn:
        assert conn.execute("SELECT model_id FROM error_model WHERE error_id=?", (err_id,)).fetchone()[0] == 9
        assert conn.execute("SELECT bot_id FROM error_bot WHERE error_id=?", (err_id,)).fetchone()[0] == bid
        assert conn.execute("SELECT code_id FROM error_code WHERE error_id=?", (err_id,)).fetchone()[0] == cid


def test_auto_link_logs_errors(caplog):
    class Dummy:
        def __init__(self) -> None:
            self.y_called: list[tuple[int, int]] = []

        @auto_link({"xs": "link_x", "ys": "link_y"})
        def add(self, rec_id: int, *, xs=None, ys=None) -> int:
            return rec_id

        def link_x(self, record_id: int, val: int) -> None:
            raise RuntimeError("boom")

        def link_y(self, record_id: int, val: int) -> None:
            self.y_called.append((record_id, val))

    d = Dummy()
    caplog.set_level(logging.ERROR)
    assert d.add(1, xs=[10], ys=[20]) == 1
    assert d.y_called == [(1, 20)]
    assert "auto link failed" in caplog.text

