import sys
import types
import io

class _FallbackResult(list):
    def __init__(self, reason: str, hits: list):
        super().__init__(hits)
        self.reason = reason

class _VectorServiceError(Exception):
    pass

class _DummyContextBuilder:
    def build(self, desc, session_id=None, include_vectors=False):
        return ("ctx", session_id or "s", [("o", "v", 0.1)])

class _DummyEmbeddingBackfill:
    def run(self, session_id="cli", db=None, batch_size=None, backend=None):
        pass

def _load_cli(monkeypatch):
    vs = types.ModuleType("vector_service")
    vs.Retriever = object
    vs.FallbackResult = _FallbackResult
    vs.ContextBuilder = _DummyContextBuilder
    vs.VectorServiceError = _VectorServiceError
    vs.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vector_service", vs)
    monkeypatch.setitem(
        sys.modules,
        "vector_service.embedding_backfill",
        types.SimpleNamespace(EmbeddingBackfill=_DummyEmbeddingBackfill),
    )
    monkeypatch.setitem(
        sys.modules,
        "vector_service.exceptions",
        types.SimpleNamespace(VectorServiceError=_VectorServiceError),
    )
    monkeypatch.setitem(
        sys.modules,
        "code_database",
        types.SimpleNamespace(PatchHistoryDB=object, CodeDB=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "patch_provenance",
        types.SimpleNamespace(
            build_chain=lambda *a, **k: [],
            search_patches_by_vector=lambda *a, **k: [],
            search_patches_by_license=lambda *a, **k: [],
            get_patch_provenance=lambda pid: [{"id": pid}],
            PatchLogger=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "quick_fix_engine",
        types.SimpleNamespace(generate_patch=lambda *a, **k: None),
    )
    import importlib
    menace_cli = importlib.import_module("menace_cli")
    return menace_cli

def _cleanup_cli():
    sys.modules.pop("menace_cli", None)


def test_patch_command(monkeypatch, tmp_path):
    menace_cli = _load_cli(monkeypatch)

    calls = {}

    def fake_generate_patch(module, *, context_builder, engine, description, **kw):
        calls["module"] = module
        calls["builder"] = context_builder
        calls["engine"] = engine
        calls["description"] = description
        return 42

    monkeypatch.setattr(
        sys.modules["quick_fix_engine"], "generate_patch", fake_generate_patch
    )

    mod = tmp_path / "m.py"
    mod.write_text("x=1")

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    out_buf = sys.stdout

    res = menace_cli.main(["patch", str(mod), "--desc", "fix it"])
    assert res == 0
    assert out_buf.getvalue().strip() == "42"
    assert calls["module"] == str(mod)
    assert calls["engine"] is None
    assert calls["description"] == "fix it"
    assert isinstance(calls["builder"], sys.modules["vector_service"].ContextBuilder)

    _cleanup_cli()


def test_embed_command(monkeypatch):
    menace_cli = _load_cli(monkeypatch)

    calls = {}
    class DummyBackfill:
        def run(self, session_id="cli", db=None, batch_size=None, backend=None):
            calls["kwargs"] = {
                "session_id": session_id,
                "db": db,
                "batch_size": batch_size,
                "backend": backend,
            }

    monkeypatch.setattr(
        sys.modules["vector_service.embedding_backfill"],
        "EmbeddingBackfill",
        lambda: DummyBackfill(),
    )
    res = menace_cli.main(
        ["embed", "--db", "code", "--batch-size", "5", "--backend", "fake"]
    )
    assert res == 0
    assert calls["kwargs"] == {
        "session_id": "cli",
        "db": "code",
        "batch_size": 5,
        "backend": "fake",
    }

    class FailBackfill:
        def run(self, session_id="cli", db=None, batch_size=None, backend=None):
            calls["kwargs"] = {
                "session_id": session_id,
                "db": db,
                "batch_size": batch_size,
                "backend": backend,
            }
            raise _VectorServiceError("boom")

    monkeypatch.setattr(
        sys.modules["vector_service.embedding_backfill"],
        "EmbeddingBackfill",
        lambda: FailBackfill(),
    )
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    res = menace_cli.main(["embed"])
    assert res == 1
    assert calls["kwargs"] == {
        "session_id": "cli",
        "db": None,
        "batch_size": None,
        "backend": None,
    }

    _cleanup_cli()


def test_new_db_command(monkeypatch):
    menace_cli = _load_cli(monkeypatch)

    calls = {}
    def fake_run(cmd):
        calls["cmd"] = cmd
        return 0
    monkeypatch.setattr(menace_cli, "_run", fake_run)
    res = menace_cli.main(["new-db", "demo"])
    assert res == 0
    assert calls["cmd"][-1] == "demo"

    _cleanup_cli()
