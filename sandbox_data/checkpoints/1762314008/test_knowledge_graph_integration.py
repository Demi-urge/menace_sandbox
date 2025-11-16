import types
import sys

import pytest

pytest.importorskip("networkx")

import menace.knowledge_graph as kgm
from menace.bot_database import BotDB, BotRecord
from menace.code_database import CodeDB, CodeRecord
from menace.menace_memory_manager import MenaceMemoryManager
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome


def test_graph_integration(tmp_path):
    bdb = BotDB(tmp_path / "b.db")
    bdb.add_bot(BotRecord(name="A", tasks=["t"], dependencies=["B"]))
    bdb.add_bot(BotRecord(name="B"))

    cdb = CodeDB(tmp_path / "c.db")
    cdb.add(CodeRecord(code="x", summary="s"), bots=[1])

    mm = MenaceMemoryManager(tmp_path / "m.db")
    mm.store("A->B", "d", tags="A")

    pdb = PathwayDB(tmp_path / "p.db")
    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))

    g = kgm.KnowledgeGraph()
    g.ingest_bots(bdb)
    g.ingest_memory(mm)
    g.ingest_pathways(pdb)
    g.ingest_code(cdb, bdb)
    g.link_pathway_to_memory(pdb, mm)
    related = g.related("bot:A", depth=3)
    assert "bot:B" in related
    assert any(r.startswith("code:") for r in related)
    assert "memory:A->B" in related
    assert "pathway:A->B" in related
