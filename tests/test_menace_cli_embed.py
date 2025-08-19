import sys
import types
import importlib


def _load_cli(monkeypatch, backfill_impl):
    sys.modules.pop("menace_cli", None)
    vs = types.ModuleType("vector_service")
    vs.__path__ = []  # type: ignore[attr-defined]
    vs.PatchLogger = object
    monkeypatch.setitem(sys.modules, "vector_service", vs)
    monkeypatch.setitem(
        sys.modules,
        "vector_service.embedding_backfill",
        types.SimpleNamespace(EmbeddingBackfill=lambda: backfill_impl),
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
            get_patch_provenance=lambda pid: [],
            PatchLogger=object,
        ),
    )
    return importlib.import_module("menace_cli")

def test_embed_single_db(monkeypatch):
    calls = {}

    class DummyBackfill:
        def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
            calls["kwargs"] = {
                "session_id": session_id,
                "dbs": dbs,
                "batch_size": batch_size,
                "backend": backend,
            }

    menace_cli = _load_cli(monkeypatch, DummyBackfill())
    res = menace_cli.main(["embed", "--db", "workflows", "--batch-size", "5", "--backend", "fake"])
    assert res == 0
    assert calls["kwargs"] == {
        "session_id": "cli",
        "dbs": ["workflows"],
        "batch_size": 5,
        "backend": "fake",
    }

def test_embed_multi_db(monkeypatch):
    calls = {}

    class DummyBackfill:
        def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
            calls["dbs"] = dbs

    menace_cli = _load_cli(monkeypatch, DummyBackfill())
    res = menace_cli.main(["embed", "--db", "code", "--db", "workflows"])
    assert res == 0
    assert calls["dbs"] == ["code", "workflows"]


def test_embed_errors(monkeypatch, capsys):
    # initialise modules to grab the custom exception class
    _ = _load_cli(monkeypatch, object())
    VecErr = sys.modules["vector_service.exceptions"].VectorServiceError

    class ErrBackfill:
        def run(self, **kwargs):
            raise VecErr("vector fail")

    menace_cli = _load_cli(monkeypatch, ErrBackfill())
    rc = menace_cli.main(["embed"])
    assert rc == 1
    assert "vector fail" in capsys.readouterr().err

    class GenBackfill:
        def run(self, **kwargs):
            raise RuntimeError("boom")

    menace_cli = _load_cli(monkeypatch, GenBackfill())
    rc = menace_cli.main(["embed"])
    assert rc == 1
    assert "boom" in capsys.readouterr().err
