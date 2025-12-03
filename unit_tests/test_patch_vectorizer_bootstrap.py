import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

def test_patch_vectorizer_bootstrap_defers_initialisation(monkeypatch, tmp_path):
    calls = {"resolve": 0, "db": 0}
    ensure_calls: list[bool] = []

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

    pv = PatchVectorizer(path=tmp_path / "patch_history.db")

    assert calls == {"resolve": 0, "db": 0}
    assert ensure_calls == []

    pv.get_vector("1")

    assert calls["resolve"] == 3  # DB, index and metadata path resolution
    assert calls["db"] == 1
    assert ensure_calls

