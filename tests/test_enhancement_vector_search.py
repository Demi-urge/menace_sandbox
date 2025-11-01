import sys
import types
import pytest

mods = {
    "menace.database_management_bot": {
        "DatabaseManagementBot": type("DatabaseManagementBot", (), {})
    },
    "menace.capital_management_bot": {
        "CapitalManagementBot": type("CapitalManagementBot", (), {})
    },
    "menace.data_bot": {"DataBot": type("DataBot", (), {})},
    "shared_gpt_memory": {
        "GPT_MEMORY_MANAGER": types.SimpleNamespace(
            conn=types.SimpleNamespace(
                execute=lambda *a, **k: types.SimpleNamespace(fetchall=lambda: [])
            )
        )
    },
    "shared_knowledge_module": {"LOCAL_KNOWLEDGE_MODULE": object()},
    "metrics_db": {
        "MetricsDB": type(
            "MetricsDB", (), {"log_embedding_staleness": staticmethod(lambda *a, **k: None)}
        )
    },
}
for name, attrs in mods.items():
    module = types.ModuleType(name)
    for attr, obj in attrs.items():
        setattr(module, attr, obj)
    sys.modules.setdefault(name, module)

import menace.chatgpt_enhancement_bot as ceb  # noqa: E402
import menace_sandbox.embeddable_db_mixin as embeddable_db_mixin  # noqa: E402

embeddable_db_mixin.MetricsDB = mods["metrics_db"]["MetricsDB"]


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = ceb.EnhancementDB(
        tmp_path / "e.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"idx.{backend}.index",
    )

    def fake_embed(text: str):
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = fake_embed

    e1 = ceb.Enhancement(
        idea="i1",
        rationale="r1",
        summary="alpha",
        before_code="a",
        after_code="b",
    )
    id1 = db.add(e1)

    assert str(id1) in db._metadata
    res1 = db.search_by_vector([1.0, 0.0], top_k=1, scope="local")
    assert res1 and res1[0].summary == "alpha"

    # insert cross-instance enhancement and embed manually
    db.conn.execute(
        (
            "INSERT INTO enhancements(idea, rationale, summary, before_code, after_code, "
            "source_menace_id) VALUES (?,?,?,?,?,?)"
        ),
        ("i2", "r2", "beta", "x", "y", "other"),
    )
    db.conn.commit()
    new_id = db.conn.execute(
        "SELECT id FROM enhancements WHERE summary='beta' AND source_menace_id=?",
        ("other",),
    ).fetchone()[0]
    assert str(new_id) not in db._metadata
    db.add_embedding(
        new_id,
        ceb.Enhancement(idea="", rationale="", summary="beta", before_code="x", after_code="y"),
        "enhancement",
        source_id=str(new_id),
    )
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1, scope="global")
    assert res2 and res2[0].summary == "beta"
    res_all = db.search_by_vector([0.0, 1.0], top_k=2, scope="all")
    assert any(r.summary == "beta" for r in res_all)

    # ensure metadata mirrored in SQLite
    row = db.conn.execute(
        "SELECT kind FROM enhancement_embeddings WHERE record_id=?", (id1,),
    ).fetchone()
    assert row and row[0] == "enhancement"
    row2 = db.conn.execute(
        "SELECT kind FROM enhancement_embeddings WHERE record_id=?", (new_id,),
    ).fetchone()
    assert row2 and row2[0] == "enhancement"
