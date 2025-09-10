import importlib
import logging
import sys
import types

import pytest

EMBEDDED: set[str] = set()


class DummyRetriever:
    def __init__(self) -> None:
        self.required_dbs: list[str] = []

    def search(self, query, top_k=5, session_id=None, max_alert_severity=1.0):
        from vector_service.retriever import FallbackResult

        missing = [db for db in self.required_dbs if db not in EMBEDDED]
        if missing:
            return FallbackResult("missing embeddings", [])
        db = self.required_dbs[0] if self.required_dbs else "db"
        return [
            {
                "origin_db": db,
                "record_id": "1",
                "score": 1.0,
                "text": "sample",
                "metadata": {"redacted": True},
            }
        ]

dummy_patch_logger = types.ModuleType("vector_service.patch_logger")
dummy_patch_logger._VECTOR_RISK = 0
sys.modules.setdefault("vector_service.patch_logger", dummy_patch_logger)


@pytest.fixture
def make_builder(monkeypatch):
    def _factory(db_name: str):
        import vector_service.context_builder as cb

        monkeypatch.setattr(cb.ContextBuilder, "refresh_db_weights", lambda self, *a, **k: None)
        monkeypatch.setattr(cb, "_ensure_vector_service", lambda: None)
        monkeypatch.setattr(cb, "ensure_embeddings_fresh", lambda dbs: None)
        dummy_patch = types.SimpleNamespace(search=lambda *a, **k: [])
        retriever = DummyRetriever()
        builder = cb.ContextBuilder(
            retriever=retriever,
            patch_retriever=dummy_patch,
            db_weights={db_name: 1.0},
        )
        retriever.required_dbs = list(builder.db_weights)
        monkeypatch.setattr(builder.patch_safety, "load_failures", lambda: None)
        return builder

    return _factory


def _load_cli(monkeypatch, db_name: str):
    sys.modules.pop("menace_cli", None)
    vs = types.ModuleType("vector_service")
    vs.__path__ = []  # type: ignore[attr-defined]
    vs.PatchLogger = object
    monkeypatch.setitem(sys.modules, "vector_service", vs)

    class _EB:
        def __init__(self, *a, **k):
            pass

        def check_out_of_sync(self, dbs=None):  # pragma: no cover - unused
            return dbs or [db_name]

        def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
            for db in dbs or []:
                EMBEDDED.add(db)

    monkeypatch.setitem(
        sys.modules,
        "vector_service.embedding_backfill",
        types.SimpleNamespace(
            EmbeddingBackfill=_EB,
            _RUN_SKIPPED=types.SimpleNamespace(
                labels=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None)
            ),
            _log_violation=lambda *a, **k: None,
            _load_registry=lambda: {db_name: ("m", "C")},
            KNOWN_DB_KINDS={db_name},
        ),
    )
    VecErr = type("VecErr", (Exception,), {})
    monkeypatch.setitem(
        sys.modules,
        "vector_service.exceptions",
        types.SimpleNamespace(VectorServiceError=VecErr),
    )
    monkeypatch.setitem(
        sys.modules,
        "code_database",
        types.SimpleNamespace(PatchHistoryDB=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "patch_provenance",
        types.SimpleNamespace(
            build_chain=lambda *a, **k: [],
            search_patches_by_vector=lambda *a, **k: [],
            search_patches_by_license=lambda *a, **k: [],
            get_patch_provenance=lambda *a, **k: [],
            PatchLogger=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "menace.plugins", types.SimpleNamespace(load_plugins=lambda sub: None)
    )
    return importlib.import_module("menace_cli")


@pytest.fixture
def embed_cli(monkeypatch):
    def _run(db_name: str):
        menace_cli = _load_cli(monkeypatch, db_name)
        assert menace_cli.main(["embed", "--db", db_name]) == 0

    return _run


@pytest.fixture(autouse=True)
def clear_embedded():
    EMBEDDED.clear()
    yield
    EMBEDDED.clear()


@pytest.mark.parametrize("db_name", ['code', 'bot'])
def test_retrieve_requires_embeddings(make_builder, embed_cli, caplog, db_name):
    builder = make_builder(db_name)
    caplog.set_level(logging.DEBUG, logger="vector_service.context_builder")

    ctx, sid, vectors = builder.build_context("query", include_vectors=True, top_k=1)
    assert vectors == []
    assert any("retriever returned fallback" in r.message for r in caplog.records)

    embed_cli(db_name)
    caplog.clear()

    ctx, sid, vectors = builder.build_context("query", include_vectors=True, top_k=1)
    assert vectors
    assert all("retriever returned fallback" not in r.message for r in caplog.records)
