import json
import sqlite3

import vector_service.patch_logger as pl_mod
from vector_service.patch_logger import PatchLogger
from vector_service.context_builder import ContextBuilder


def test_lessons_retrieved_after_patch(tmp_path, monkeypatch):
    calls: list[int] = []

    class StubResearchItem:
        def __init__(self, topic="", content="", summary="", category="", type_=""):
            self.topic = topic
            self.content = content
            self.summary = summary
            self.category = category
            self.type_ = type_

    class StubInfoDB:
        def __init__(self, path):
            self.path = path
            self.conn = sqlite3.connect(self.path)
            self.conn.execute("CREATE TABLE info(id INTEGER PRIMARY KEY, summary TEXT)")

        def add(self, item: StubResearchItem, embedding=None, *, workflows=None, enhancements=None):
            cur = self.conn.execute("INSERT INTO info(summary) VALUES (?)", (item.summary,))
            self.conn.commit()
            rid = int(cur.lastrowid)
            self.add_embedding(rid, item, "info")
            return rid

        def add_embedding(self, record_id, record, kind, *, source_id=""):
            calls.append(int(record_id))

    monkeypatch.setattr(pl_mod, "InfoDB", StubInfoDB)
    monkeypatch.setattr(pl_mod, "ResearchItem", StubResearchItem)

    monkeypatch.setattr(
        pl_mod,
        "_summarise_text",
        lambda text, ratio=0.2: "LESSON",
    )

    info_db = StubInfoDB(str(tmp_path / "info.db"))
    pl = PatchLogger(info_db=info_db)

    class DummyRetriever:
        def __init__(self, db):
            self.db = db

        def search(self, query, **kwargs):
            rows = self.db.conn.execute("SELECT id, summary FROM info").fetchall()
            hits = []
            for rid, summary in rows:
                if query in summary:
                    hits.append(
                        {
                            "origin_db": "information",
                            "record_id": rid,
                            "score": 1.0,
                            "metadata": {
                                "summary": summary,
                                "content": summary,
                                "lessons": summary,
                            },
                        }
                    )
            return hits

    cb = ContextBuilder(retriever=DummyRetriever(info_db))
    assert json.loads(cb.build("LESSON")) == {}

    vec_id = "code:1"
    meta = {vec_id: {"code": "print('x')"}}
    pl.track_contributors([vec_id], True, retrieval_metadata=meta)

    assert calls  # embedding added

    cb = ContextBuilder(retriever=DummyRetriever(info_db))
    ctx = json.loads(cb.build("LESSON"))
    assert ctx["information"][0]["desc"] == "LESSON"
