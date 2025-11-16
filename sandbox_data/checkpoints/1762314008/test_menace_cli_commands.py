import sys
import types
import io
import json

class _FallbackResult(list):
    def __init__(self, reason: str, hits: list):
        super().__init__(hits)
        self.reason = reason

class _VectorServiceError(Exception):
    pass

class _DummyContextBuilder:
    def __init__(self, retriever=None, **kwargs):
        self.retriever = retriever

    def refresh_db_weights(self):
        return None

    def build(self, desc, session_id=None, include_vectors=False):
        return ("ctx", session_id or "s", [("o", "v", 0.1)])

class _DummyEmbeddingBackfill:
    def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
        pass

def _load_cli(monkeypatch):
    sys.modules.pop("menace_cli", None)
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
    class DummyPatchDB:
        def __init__(self):
            self.records = {}
            self.next_id = 1
            DummyPatchDB.instance = self

        def add(self, rec):
            pid = self.next_id
            self.next_id += 1
            self.records[pid] = rec
            return pid

        def get(self, pid):
            return self.records.get(pid)

        def list_patches(self, limit):  # pragma: no cover - not used
            return []

    monkeypatch.setitem(
        sys.modules,
        "code_database",
        types.SimpleNamespace(PatchHistoryDB=DummyPatchDB, CodeDB=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "patch_provenance",
        types.SimpleNamespace(
            build_chain=lambda *a, **k: [],
            search_patches_by_vector=lambda *a, **k: [],
            search_patches_by_license=lambda *a, **k: [],
            PatchLogger=lambda *a, **k: object(),
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

    def fake_generate_patch(
        module, manager, *, context_builder, engine, description, patch_logger=None, **kw
    ):
        calls["module"] = module
        calls["builder"] = context_builder
        calls["engine"] = engine
        calls["description"] = description
        db = sys.modules["code_database"].PatchHistoryDB.instance
        pid = db.add(types.SimpleNamespace(filename=module))
        return pid

    monkeypatch.setattr(
        sys.modules["quick_fix_engine"], "generate_patch", fake_generate_patch
    )

    mod = tmp_path / "m.py"  # path-ignore
    mod.write_text("x=1")

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    out_buf = sys.stdout

    res = menace_cli.main(["patch", str(mod), "--desc", "fix it"])
    assert res == 0
    assert json.loads(out_buf.getvalue()) == {
        "patch_id": 1,
        "files": [str(mod)],
    }
    assert calls["module"] == str(mod)
    assert calls["engine"] is None
    assert calls["description"] == "fix it"
    assert isinstance(calls["builder"], sys.modules["vector_service"].ContextBuilder)

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
