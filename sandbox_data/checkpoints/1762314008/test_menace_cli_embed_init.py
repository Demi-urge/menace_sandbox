import os
import sqlite3
import sys
import types
import importlib


def _make_dbs(tmp_path, monkeypatch):
    paths = {}
    for name in ("code", "bot", "error", "workflow"):
        db_path = tmp_path / f"{name}.db"
        paths[name] = str(db_path)
        env = f"{name.upper()}_DB_PATH"
        if name == "workflow":
            env = "WORKFLOWS_DB_PATH"
        monkeypatch.setenv(env, str(db_path))
    return paths


def _setup_dummy_registry(monkeypatch):
    import vector_service.embedding_backfill as eb

    EMBED_MAPS: dict[str, list[str]] = {}

    def _make_cls(env_var, cls_name):
        class _DB(eb.EmbeddableDBMixin):
            def __init__(self, vector_backend: str | None = None) -> None:
                self.path = os.environ[env_var]
                self.conn = sqlite3.connect(self.path)  # noqa: SQL001
                self.conn.execute("CREATE TABLE IF NOT EXISTS data(id TEXT, txt TEXT)")
                if not self.conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]:
                    self.conn.executemany(
                        "INSERT INTO data(id, txt) VALUES (?, ?)",
                        [("1", "a"), ("2", "b")],
                    )
                self._id_map = EMBED_MAPS.setdefault(self.path, [])

            def iter_records(self):
                cur = self.conn.execute("SELECT id, txt FROM data")
                for rid, txt in cur:
                    yield rid, txt, "text"

            def add_embedding(self, record_id, record, kind, **kwargs):  # type: ignore[override]
                rid = str(record_id)
                if rid not in self._id_map:
                    self._id_map.append(rid)

            def needs_refresh(self, record_id, record=None):  # type: ignore[override]
                return str(record_id) not in self._id_map

            def vector(self, record):  # type: ignore[override]
                return [1.0]

            def backfill_embeddings(self, batch_size: int = 100) -> None:  # type: ignore[override]
                for rid, rec, kind in self.iter_records():
                    self.add_embedding(rid, rec, kind)

        _DB.__name__ = cls_name
        return _DB

    mod = types.ModuleType("dummy_dbs")
    mod.CodeDB = _make_cls("CODE_DB_PATH", "CodeDB")
    mod.BotDB = _make_cls("BOT_DB_PATH", "BotDB")
    mod.ErrorDB = _make_cls("ERROR_DB_PATH", "ErrorDB")
    mod.WorkflowDB = _make_cls("WORKFLOWS_DB_PATH", "WorkflowDB")
    monkeypatch.setitem(sys.modules, "dummy_dbs", mod)

    def _reg(path=None):
        return {
            "code": ("dummy_dbs", "CodeDB"),
            "bot": ("dummy_dbs", "BotDB"),
            "error": ("dummy_dbs", "ErrorDB"),
            "workflow": ("dummy_dbs", "WorkflowDB"),
        }

    class DummyBackfill:
        def __init__(self, *a, **k):
            self.backend = "annoy"

        def _load_known_dbs(self, names=None):
            mapping = {
                "code": mod.CodeDB,
                "bot": mod.BotDB,
                "error": mod.ErrorDB,
                "workflow": mod.WorkflowDB,
            }
            if names:
                return [mapping[n] for n in names if n in mapping]
            return list(mapping.values())

        def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
            for cls in self._load_known_dbs(dbs):
                db = cls()
                for rid, rec, kind in db.iter_records():
                    db.add_embedding(rid, rec, kind)
                db.conn.close()

    monkeypatch.setattr(eb, "EmbeddingBackfill", DummyBackfill)
    monkeypatch.setattr(eb, "_load_registry", _reg)
    monkeypatch.setattr(
        eb,
        "_RUN_SKIPPED",
        types.SimpleNamespace(
            labels=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None)
        ),
    )
    monkeypatch.setattr(eb, "_log_violation", lambda *a, **k: None)


def _load_cli(monkeypatch):
    sys.modules.pop("menace_cli", None)
    monkeypatch.setitem(sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object))
    monkeypatch.setitem(
        sys.modules,
        "cache_utils",
        types.SimpleNamespace(
            get_cached_chain=lambda *a, **k: [],
            set_cached_chain=lambda *a, **k: None,
            _get_cache=lambda: {},
            clear_cache=lambda: None,
            show_cache=lambda: {},
            cache_stats=lambda: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "workflow_synthesizer_cli",
        types.SimpleNamespace(run=lambda *a, **k: 0),
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
    monkeypatch.setitem(
        sys.modules,
        "menace.plugins",
        types.SimpleNamespace(load_plugins=lambda sub: None),
    )
    return importlib.import_module("menace_cli")


def test_embed_init_counts(tmp_path, monkeypatch, capsys):
    _make_dbs(tmp_path, monkeypatch)
    _setup_dummy_registry(monkeypatch)
    menace_cli = _load_cli(monkeypatch)
    rc = menace_cli.main(["embed", "init"])
    assert rc == 0
    mod = sys.modules["dummy_dbs"]
    for cls_name in ("CodeDB", "BotDB", "ErrorDB", "WorkflowDB"):
        db = getattr(mod, cls_name)()
        record_total = sum(1 for _ in db.iter_records())
        assert record_total == len(db._id_map)
