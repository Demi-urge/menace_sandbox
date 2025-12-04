import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path


class _StubEmbeddableDBMixin:
    def __init__(self, *args, **kwargs):
        self._index_loaded = False

    def _ensure_index_loaded(self):
        self._index_loaded = True

    def get_vector(self, _record_id):  # pragma: no cover - stubbed out for tests
        self._ensure_index_loaded()
        return None


edm = ModuleType("embeddable_db_mixin")
edm.EmbeddableDBMixin = _StubEmbeddableDBMixin
sys.modules.setdefault("embeddable_db_mixin", edm)
sys.modules.setdefault("menace_sandbox.embeddable_db_mixin", edm)

import_compat_stub = ModuleType("import_compat")
import_compat_stub.bootstrap = lambda *_a, **_k: None
def _missing(name: str):
    err = ModuleNotFoundError(str(name))
    err.name = str(name)
    raise err


import_compat_stub.load_internal = _missing

sys.modules.setdefault("import_compat", import_compat_stub)
menace_pkg = sys.modules.setdefault("menace_sandbox", ModuleType("menace_sandbox"))
menace_pkg.import_compat = import_compat_stub
sys.modules.setdefault("menace_sandbox.import_compat", import_compat_stub)

dpr = ModuleType("dynamic_path_router")
dpr.get_project_root = lambda: Path(".")
dpr.resolve_path = lambda path: str(Path(path))
sys.modules.setdefault("dynamic_path_router", dpr)


class _StubPatchHistoryDB:
    def __init__(self, path, *, bootstrap_fast=None):
        self.path = Path(path) if path is not None else Path("patch_history.db")
        self.router = SimpleNamespace(
            get_connection=lambda *_a, **_k: SimpleNamespace(
                execute=lambda *_x, **_y: SimpleNamespace(fetchall=lambda: [])
            )
        )

    def get(self, _):
        return None


code_db_stub = ModuleType("code_database")
code_db_stub.PatchHistoryDB = _StubPatchHistoryDB
sys.modules.setdefault("code_database", code_db_stub)
sys.modules.setdefault("menace_sandbox.code_database", code_db_stub)

def test_patch_vectorizer_bootstrap_defers_initialisation(monkeypatch, tmp_path):
    calls = {"resolve": 0, "db": 0}
    ensure_calls: list[bool] = []
    scheduled: list[str | None] = []

    import embeddable_db_mixin as edm

    sandbox_stub = ModuleType("menace_sandbox")
    sandbox_stub.embeddable_db_mixin = edm
    monkeypatch.setitem(sys.modules, "menace_sandbox", sandbox_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.embeddable_db_mixin", edm)
    monkeypatch.setitem(sys.modules, "embeddable_db_mixin", edm)

    def fake_resolve(path: str) -> str:
        calls["resolve"] += 1
        return path

    class DummyDB:
        def __init__(self, path, *, bootstrap_fast=None):
            calls["db"] += 1
            self.path = Path(path) if path is not None else tmp_path / "patch_history.db"
            self.router = SimpleNamespace(
                get_connection=lambda *_a, **_k: SimpleNamespace(
                    execute=lambda *_x, **_y: SimpleNamespace(fetchall=lambda: [])
                )
            )

        def get(self, _):
            return None

    from vector_service.patch_vectorizer import EmbeddableDBMixin, PatchVectorizer

    original_ensure = EmbeddableDBMixin._ensure_index_loaded

    def track_ensure(self):
        ensure_calls.append(True)
        return original_ensure(self)

    monkeypatch.setenv("MENACE_BOOTSTRAP_FAST", "1")
    monkeypatch.setattr("vector_service.patch_vectorizer.resolve_path", fake_resolve)
    monkeypatch.setattr("vector_service.patch_vectorizer.PatchHistoryDB", DummyDB)
    monkeypatch.setattr(EmbeddableDBMixin, "_ensure_index_loaded", track_ensure)

    def fake_activate_async(self, *, reason=None):
        scheduled.append(reason)
        return None

    monkeypatch.setattr(PatchVectorizer, "activate_async", fake_activate_async, raising=False)
    monkeypatch.setattr(PatchVectorizer, "_bootstrap_background_executor", lambda self: None)

    pv = PatchVectorizer(path=tmp_path / "patch_history.db")

    assert calls == {"resolve": 0, "db": 0}
    assert ensure_calls == []

    assert pv.get_vector("1") is None
    assert list(pv.iter_records()) == []

    assert calls == {"resolve": 0, "db": 0}
    assert ensure_calls == []
    assert pv._bootstrap_deferral_reason in {"index_load", "iter_records"}
    assert not pv._bootstrap_deferral_scheduled
    assert scheduled and scheduled[0] == "index_load"


def test_patch_vectorizer_bootstrap_schedules_background(monkeypatch, tmp_path):
    submitted = {}

    import embeddable_db_mixin as edm

    sandbox_stub = ModuleType("menace_sandbox")
    sandbox_stub.embeddable_db_mixin = edm
    monkeypatch.setitem(sys.modules, "menace_sandbox", sandbox_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.embeddable_db_mixin", edm)
    monkeypatch.setitem(sys.modules, "embeddable_db_mixin", edm)

    def fake_resolve(path: str) -> str:
        return path

    class DummyDB:
        def __init__(self, path, *, bootstrap_fast=None):
            self.path = Path(path) if path is not None else tmp_path / "patch_history.db"
            self.router = SimpleNamespace(
                get_connection=lambda *_a, **_k: SimpleNamespace(
                    execute=lambda *_x, **_y: SimpleNamespace(fetchall=lambda: [])
                )
            )

        def get(self, _):
            return None

    class DummyExecutor:
        def submit(self, func):
            submitted["func"] = func
            return None

    from vector_service.patch_vectorizer import PatchVectorizer

    monkeypatch.setenv("MENACE_BOOTSTRAP_FAST", "1")
    monkeypatch.setattr("vector_service.patch_vectorizer.resolve_path", fake_resolve)
    monkeypatch.setattr("vector_service.patch_vectorizer.PatchHistoryDB", DummyDB)
    monkeypatch.setattr(
        PatchVectorizer, "_bootstrap_background_executor", lambda self: DummyExecutor()
    )

    pv = PatchVectorizer(path=tmp_path / "patch_history.db")

    assert pv.get_vector("1") is None
    assert submitted["func"]
    assert pv._bootstrap_deferral_reason == "index_load"
    assert pv._bootstrap_deferral_scheduled

