import os
import sys
import types
import importlib.util
import logging
from pathlib import Path
import pytest
import dynamic_path_router

if not hasattr(dynamic_path_router, "clear_cache"):
    def _resolve_path(p: str) -> Path:
        root = Path(os.environ.get("SANDBOX_REPO_PATH", "."))
        cand = root / p
        if not cand.exists():
            raise FileNotFoundError(p)
        return cand

    dynamic_path_router.resolve_path = _resolve_path  # type: ignore[attr-defined]
    dynamic_path_router.path_for_prompt = lambda p: Path(p).as_posix()  # type: ignore[attr-defined]
    dynamic_path_router.clear_cache = lambda: None  # type: ignore[attr-defined]

# Avoid heavy imports from the real package
package = types.ModuleType("menace")
package.__path__ = []
sys.modules["menace"] = package

# Stub required submodules
error_bot = types.ModuleType("menace.error_bot")
error_bot.ErrorDB = object
sys.modules["menace.error_bot"] = error_bot

scm = types.ModuleType("menace.self_coding_manager")
scm.SelfCodingManager = object
sys.modules["menace.self_coding_manager"] = scm

# Stubs for modules imported by patch_provenance/code_database


def _auto_link(*a, **k):
    def decorator(func):
        return func

    return decorator


sys.modules.setdefault("auto_link", types.SimpleNamespace(auto_link=_auto_link))
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(
        publish_with_retry=lambda *a, **k: None,
        with_retry=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(
        send_discord_alert=lambda *a, **k: None,
        CONFIG={},
    ),
)

kg = types.ModuleType("menace.knowledge_graph")
kg.KnowledgeGraph = object
sys.modules["menace.knowledge_graph"] = kg

# Minimal vector_service stubs


class _DummyContextBuilder:
    def __init__(self, *a, retriever=None, **k):
        self.retriever = retriever

    def build(self, *a, **k):
        return ""

    def refresh_db_weights(self):
        return None


vec = types.SimpleNamespace(
    ContextBuilder=_DummyContextBuilder,
    Retriever=object,
    FallbackResult=list,
    EmbeddingBackfill=object,
)
sys.modules.setdefault("vector_service", vec)
sys.modules.setdefault("vector_service.patch_logger", types.SimpleNamespace(PatchLogger=object))
sys.modules.setdefault("vector_service.context_builder", vec)
sc_stub = types.SimpleNamespace(compress_snippets=lambda meta, **k: meta)
sys.modules.setdefault("snippet_compressor", sc_stub)
sys.modules.setdefault("menace.snippet_compressor", sc_stub)
sys.modules.setdefault(
    "menace.codebase_diff_checker",
    types.SimpleNamespace(generate_code_diff=lambda *a, **k: "", flag_risky_changes=lambda *a, **k: False),
)

# Load QuickFixEngine without importing the full package
spec = importlib.util.spec_from_file_location(
    "menace.quick_fix_engine",
    dynamic_path_router.path_for_prompt("quick_fix_engine.py"),  # path-ignore
)
quick_fix = importlib.util.module_from_spec(spec)
sys.modules["menace.quick_fix_engine"] = quick_fix
spec.loader.exec_module(quick_fix)
QuickFixEngine = quick_fix.QuickFixEngine


class DummyManager:
    def run_patch(self, path, desc):
        self.calls = getattr(self, "calls", [])
        self.calls.append((path, desc))


class FailingGraph:
    def add_telemetry_event(self, *a, **k):
        raise RuntimeError("boom")

    def update_error_stats(self, *a, **k):
        pass


def test_telemetry_error_logged(monkeypatch, tmp_path, caplog):
    engine = QuickFixEngine(
        error_db=None,
        manager=DummyManager(),
        threshold=1,
        graph=FailingGraph(),
        context_builder=quick_fix.ContextBuilder(),
    )
    (tmp_path / "bot.py").write_text("x = 1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "bot", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        engine.run("bot")
    assert "telemetry update failed" in caplog.text


class DummyErrorDB:
    def top_error_module(self, bot):
        return ("runtime_fault", "b", {"a": 1, "b": 2}, 2, bot)


class DummyGraph:
    def __init__(self):
        self.events = []
        self.updated = None

    def add_telemetry_event(self, *a, **k):
        self.events.append((a, k))

    def update_error_stats(self, db):
        self.updated = db


def test_run_targets_frequent_module(tmp_path, monkeypatch):
    engine = QuickFixEngine(
        error_db=DummyErrorDB(),
        manager=DummyManager(),
        threshold=2,
        graph=DummyGraph(),
        context_builder=quick_fix.ContextBuilder(),
    )
    (tmp_path / "b.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    engine.run("bot")
    assert engine.manager.calls[0][0] == dynamic_path_router.resolve_path("b.py")  # path-ignore
    assert engine.graph.events[0][0][2] == "b"
    assert engine.graph.events[0][1]["resolved"] is True
    assert engine.graph.updated is engine.db


class DummyPreemptiveDB:
    def __init__(self):
        self.records = []

    def log_preemptive_patch(self, module, risk, patch_id):
        self.records.append((module, risk, patch_id))


class DummyResult:
    def __init__(self, patch_id):
        self.patch_id = patch_id


class DummyManager2:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def run_patch(self, path, desc):
        self.calls.append((path, desc))
        if self.fail:
            raise RuntimeError("boom")
        return DummyResult(123)


def test_preemptive_patch_modules(tmp_path, monkeypatch):
    db = DummyPreemptiveDB()
    mgr = DummyManager2()
    engine = QuickFixEngine(
        error_db=db,
        manager=mgr,
        threshold=0,
        graph=None,
        context_builder=quick_fix.ContextBuilder(),
    )
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    modules = [("mod", 0.9), ("low", 0.1)]
    engine.preemptive_patch_modules(modules, risk_threshold=0.5)
    assert mgr.calls == [
        (dynamic_path_router.resolve_path("mod.py"), "preemptive_patch")
    ]  # path-ignore
    assert db.records == [("mod", 0.9, 123)]


def test_preemptive_patch_falls_back(monkeypatch, tmp_path):
    db = DummyPreemptiveDB()
    mgr = DummyManager2(fail=True)
    engine = QuickFixEngine(
        error_db=db,
        manager=mgr,
        threshold=0,
        graph=None,
        context_builder=quick_fix.ContextBuilder(),
    )
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setattr(
        quick_fix, "generate_patch", lambda m, engine=None, **kw: 999
    )
    engine.preemptive_patch_modules([("mod", 0.8)], risk_threshold=0.5)
    assert mgr.calls == [
        (dynamic_path_router.resolve_path("mod.py"), "preemptive_patch")
    ]  # path-ignore
    assert db.records == [("mod", 0.8, 999)]


def test_generate_patch_blocks_risky(monkeypatch, tmp_path):
    path = tmp_path / "a.py"  # path-ignore
    path.write_text("x=1\n")

    class DummyEngine:
        def apply_patch(self, p, *a, **k):
            with open(p, "a", encoding="utf-8") as f:
                f.write("eval('2')\n")
            return 1, "", ""
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    res = quick_fix.generate_patch(
        str(path), engine=DummyEngine(), context_builder=quick_fix.ContextBuilder()
    )
    assert res is None
    assert path.read_text() == "x=1\n"


def test_run_records_retrieval_metadata(tmp_path, monkeypatch):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    from code_database import PatchHistoryDB, PatchRecord
    from patch_provenance import get_patch_provenance
    from vector_service.patch_logger import PatchLogger

    db = PatchHistoryDB()
    patch_id = db.add(PatchRecord("mod.py", "desc", 1.0, 2.0))  # path-ignore

    class Manager:
        def run_patch(self, path, desc, context_meta=None):
            return types.SimpleNamespace(patch_id=patch_id)

    class Retriever:
        def search(self, query, top_k, session_id):
            return [
                {
                    "origin_db": "db1",
                    "record_id": "vec1",
                    "score": 0.5,
                    "license": "mit",
                    "semantic_alerts": ["unsafe"],
                }
            ]

    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
        patch_logger=PatchLogger(patch_db=db),
        context_builder=quick_fix.ContextBuilder(),
    )
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    engine.run("bot")
    prov = get_patch_provenance(patch_id, patch_db=db)
    assert prov[0]["license"] == "mit"
    assert prov[0]["semantic_alerts"] == ["unsafe"]


def test_run_records_ancestry_without_logger(tmp_path, monkeypatch):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    from code_database import PatchHistoryDB, PatchRecord
    from patch_provenance import get_patch_provenance

    db = PatchHistoryDB()
    patch_id = db.add(PatchRecord("mod.py", "desc", 1.0, 2.0))  # path-ignore

    class Manager:
        def __init__(self):
            self.engine = types.SimpleNamespace(patch_db=db)

        def run_patch(self, path, desc, context_meta=None):
            return types.SimpleNamespace(patch_id=patch_id)

    class Retriever:
        def search(self, query, top_k, session_id):
            return [{"origin_db": "db1", "record_id": "vec1", "score": 0.4}]

    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
        context_builder=quick_fix.ContextBuilder(),
    )
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)

    engine.run("bot")
    prov = get_patch_provenance(patch_id, patch_db=db)
    assert prov[0]["vector_id"] == "vec1"


def test_generate_patch_resolves_module_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    nested = repo / "pkg"
    nested.mkdir()
    mod = nested / "mod.py"  # path-ignore
    mod.write_text("x=1\n")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    dynamic_path_router.clear_cache()

    class DummyEngine:
        patch_db = None

        def __init__(self):
            self.calls = []

        def apply_patch(self, path, *a, **k):
            self.calls.append(path)
            return 1, "", 0.0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda a, b: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda d: {})
    monkeypatch.setattr(quick_fix, "_collect_diff_data", lambda a, b: {})
    monkeypatch.setattr(
        quick_fix,
        "HumanAlignmentAgent",
        lambda: types.SimpleNamespace(evaluate_changes=lambda *a, **k: {}),
    )
    monkeypatch.setattr(quick_fix, "log_violation", lambda *a, **k: None)
    monkeypatch.setattr(
        quick_fix,
        "EmbeddingBackfill",
        lambda: types.SimpleNamespace(run=lambda *a, **k: None),
    )
    sys.modules.setdefault(
        "sandbox_runner", types.SimpleNamespace(post_round_orphan_scan=lambda p: None)
    )

    engine = DummyEngine()
    patch_id = quick_fix.generate_patch(
        "pkg/mod", engine=engine, context_builder=quick_fix.ContextBuilder()
    )
    assert patch_id == 1
    assert engine.calls and engine.calls[0] == mod


def test_generate_patch_uses_context_builder(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    mod = repo / "mod.py"  # path-ignore
    mod.write_text("x=1\n")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    dynamic_path_router.clear_cache()

    captured: dict[str, object] = {}

    class DummyBuilder:
        def __init__(self):
            self.refreshed = False

        def refresh_db_weights(self):
            self.refreshed = True
            return None

        def build(self, desc, session_id=None, include_vectors=False):
            captured["build_args"] = (desc, session_id, include_vectors)
            return "snippet"

    class DummyEngine:
        def apply_patch(self, path, desc, **kw):
            captured["patched_desc"] = desc
            return 1, "", 0.0

    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda a, b: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda d: {})
    monkeypatch.setattr(quick_fix, "_collect_diff_data", lambda a, b: {})
    monkeypatch.setattr(
        quick_fix,
        "HumanAlignmentAgent",
        lambda: types.SimpleNamespace(evaluate_changes=lambda *a, **k: {}),
    )
    monkeypatch.setattr(quick_fix, "log_violation", lambda *a, **k: None)
    monkeypatch.setattr(
        quick_fix,
        "EmbeddingBackfill",
        lambda: types.SimpleNamespace(run=lambda *a, **k: None),
    )
    sys.modules.setdefault(
        "sandbox_runner", types.SimpleNamespace(post_round_orphan_scan=lambda p: None)
    )

    engine = DummyEngine()
    builder = DummyBuilder()
    patch_id = quick_fix.generate_patch(
        "mod", engine=engine, context_builder=builder, description="fix bug"
    )
    assert patch_id == 1
    build_args = captured.get("build_args")
    assert build_args and build_args[0] == "fix bug"
    assert build_args[2] is True
    assert "snippet" in captured.get("patched_desc", "")
    assert builder.refreshed is True


def test_init_auto_builds_context_builder(tmp_path, monkeypatch):
    class BuildCapturingBuilder:
        def __init__(self, *a, retriever=None, **k):
            self.retriever = retriever
            self.calls = []

        def build(self, query, session_id=None, **kw):
            self.calls.append((query, session_id))
            return ""

    class Retriever:
        pass

    retriever = Retriever()
    builder = BuildCapturingBuilder(retriever=retriever)
    engine = QuickFixEngine(
        error_db=DummyErrorDB(),
        manager=DummyManager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=retriever,
        context_builder=builder,
    )
    assert engine.context_builder is builder

    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    engine.run("bot")
    assert engine.context_builder.calls


def test_generate_patch_context_compression(monkeypatch, tmp_path):
    class SentinelBuilder(_DummyContextBuilder):
        def __init__(self, **dbs):
            self.dbs = dbs

        def build(self, *_, **__):
            return "RAW-" + ",".join(sorted(self.dbs.values()))

    def fake_compress(meta, **_):
        txt = meta.get("snippet", "")
        return {"snippet": txt.replace("RAW-", "COMPRESSED-")}

    monkeypatch.setattr(quick_fix, "compress_snippets", fake_compress)

    builder = SentinelBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )

    class DummyEngine:
        def __init__(self):
            self.descs = []

        def apply_patch_with_retry(self, path, desc, **_):
            self.descs.append(desc)
            return 1, "", ""

    eng = DummyEngine()
    p = tmp_path / "mod.py"
    p.write_text("print(1)\n")  # path-ignore
    quick_fix.generate_patch(p.as_posix(), engine=eng, context_builder=builder)
    assert eng.descs and "COMPRESSED-bots.db,code.db,errors.db,workflows.db" in eng.descs[0]
    assert "RAW-bots.db,code.db,errors.db,workflows.db" not in eng.descs[0]

    with pytest.raises(TypeError):
        quick_fix.generate_patch(p.as_posix(), engine=eng)  # type: ignore[call-arg]


def test_run_context_compression(monkeypatch, tmp_path):
    class Builder(_DummyContextBuilder):
        def build(self, *_, **__):
            return "RAW-snippet"

    def fake_compress(meta, **_):
        txt = meta.get("snippet", "")
        return {"snippet": txt.replace("RAW", "COMPRESSED")}

    monkeypatch.setattr(quick_fix, "compress_snippets", fake_compress)

    manager = DummyManager()
    engine = QuickFixEngine(
        error_db=DummyErrorDB(),
        manager=manager,
        threshold=1,
        graph=DummyGraph(),
        context_builder=Builder(),
    )
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dynamic_path_router.clear_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)

    engine.run("bot")
    desc = manager.calls[0][1]
    assert "COMPRESSED-snippet" in desc
    assert "RAW-snippet" not in desc
