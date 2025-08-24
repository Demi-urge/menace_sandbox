import db_router
import sys
import types

# Minimal stubs to avoid heavy optional dependencies during import


class _EmbeddableStub:
    def __init__(self, *a, **k):
        pass

    def add_embedding(self, *a, **k):  # pragma: no cover - simple stub
        pass

    def encode_text(self, text):  # pragma: no cover - simple stub
        return [0.0]


sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(EmbeddableDBMixin=_EmbeddableStub, CognitionLayer=object),
)
sys.modules.setdefault(
    "auto_link", types.SimpleNamespace(auto_link=lambda *_a, **_k: (lambda f: f))
)
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(
        publish_with_retry=lambda *a, **k: True,
        with_retry=lambda func, **k: func(),
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={})
)

import menace.code_database as cdb  # noqa: E402
from vector_service import EmbeddableDBMixin  # noqa: E402


def _setup_code_db(tmp_path):
    shared = tmp_path / "shared.db"
    # disable embedding to keep tests lightweight
    cdb.CodeDB._embed_record_on_write = lambda *a, **k: None
    router_a = db_router.init_db_router("alpha", str(tmp_path / "alpha.db"), str(shared))
    db_a = cdb.CodeDB(tmp_path / "a.db", router=router_a)
    id_a = db_a.add(cdb.CodeRecord(code="print('a')", summary="A", complexity_score=1.0))
    db_a.link_bot(id_a, "shared")

    router_b = db_router.init_db_router("beta", str(tmp_path / "beta.db"), str(shared))
    db_b = cdb.CodeDB(tmp_path / "b.db", router=router_b)
    id_b = db_b.add(cdb.CodeRecord(code="print('b')", summary="B", complexity_score=2.0))
    db_b.link_bot(id_b, "shared")

    return {
        "router_a": router_a,
        "router_b": router_b,
        "db_a": db_a,
        "db_b": db_b,
        "id_a": id_a,
        "id_b": id_b,
    }


def test_fetch_all_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert {r["summary"] for r in db_a.fetch_all(scope="local")} == {"A"}
    assert {r["summary"] for r in db_a.fetch_all(scope="global")} == {"B"}
    assert {r["summary"] for r in db_a.fetch_all(scope="all")} == {"A", "B"}


def test_by_complexity_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert [r["summary"] for r in db_a.by_complexity(scope="local")] == ["A"]
    assert [r["summary"] for r in db_a.by_complexity(scope="global")] == ["B"]
    assert {r["summary"] for r in db_a.by_complexity(scope="all")} == {"A", "B"}


def test_search_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert [r["summary"] for r in db_a.search("print", scope="local")] == ["A"]
    assert [r["summary"] for r in db_a.search("print", scope="global")] == ["B"]
    assert {r["summary"] for r in db_a.search("print", scope="all")} == {"A", "B"}


def test_search_fallback_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert [r["summary"] for r in db_a.search_fallback("print", scope="local")] == ["A"]
    assert [r["summary"] for r in db_a.search_fallback("print", scope="global")] == ["B"]
    assert {r["summary"] for r in db_a.search_fallback("print", scope="all")} == {"A", "B"}


def test_codes_for_bot_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    ids_local = set(db_a.codes_for_bot("shared", scope="local"))
    ids_global = set(db_a.codes_for_bot("shared", scope="global"))
    ids_all = set(db_a.codes_for_bot("shared", scope="all"))
    assert ids_all == ids_local | ids_global


def test_add_embedding_scopes(tmp_path, monkeypatch):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]
    id_a = ctx["id_a"]
    id_b = ctx["id_b"]

    calls: list[int] = []

    def fake_add(self, record_id, record, kind, *, source_id=""):
        calls.append(int(record_id))

    monkeypatch.setattr(EmbeddableDBMixin, "add_embedding", fake_add)

    db_a.add_embedding(id_a, object(), "code", scope="local")
    db_a.add_embedding(id_b, object(), "code", scope="local")
    db_a.add_embedding(id_b, object(), "code", scope="global")
    db_a.add_embedding(id_a, object(), "code", scope="all")
    db_a.add_embedding(id_b, object(), "code", scope="all")

    assert calls == [id_a, id_b, id_a, id_b]


def test_iter_records_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    local = {rec["summary"] for _, rec, _ in db_a.iter_records(scope="local")}
    global_ = {rec["summary"] for _, rec, _ in db_a.iter_records(scope="global")}
    all_ = {rec["summary"] for _, rec, _ in db_a.iter_records(scope="all")}

    assert local == {"A"}
    assert global_ == {"B"}
    assert all_ == {"A", "B"}

def test_by_complexity_min_score_scope(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]
    # min_score above local record complexity returns no local rows
    assert db_a.by_complexity(min_score=1.5, scope="local") == []
    assert [r["summary"] for r in db_a.by_complexity(min_score=1.5, scope="global")] == ["B"]
    assert [r["summary"] for r in db_a.by_complexity(min_score=1.5, scope="all")] == ["B"]


def test_search_fallback_scope_filters(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]
    # term matches only the global record
    assert db_a.search_fallback("b", scope="local") == []
    assert [r["summary"] for r in db_a.search_fallback("b", scope="global")] == ["B"]
    assert [r["summary"] for r in db_a.search_fallback("b", scope="all")] == ["B"]

