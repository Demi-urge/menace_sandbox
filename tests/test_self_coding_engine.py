from pathlib import Path
import json
import sys
import types

# Lightweight stub for vector_service to avoid heavy imports
vec_mod = types.ModuleType("vector_service")


class _VSError(Exception):
    pass


vec_mod.ContextBuilder = object  # type: ignore[attr-defined]
vec_mod.ErrorResult = object  # type: ignore[attr-defined]
vec_mod.PatchLogger = object  # type: ignore[attr-defined]
vec_mod.VectorServiceError = _VSError
sys.modules.setdefault("vector_service", vec_mod)

from vector_service import VectorServiceError

sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(CodeDB=object))

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
stub = types.ModuleType("stub")
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("run_autonomous", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None))
sys.modules.setdefault(
    "menace.run_autonomous", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)

import menace.self_coding_engine as sce  # noqa: E402
import menace.code_database as cd  # noqa: E402
import menace.menace_memory_manager as mm  # noqa: E402
import menace.data_bot as db  # noqa: E402


def test_apply_patch_reverts_on_complexity(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, data_bot=data_bot, patch_db=patch_db)

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_x():\n    pass\n")

    path = tmp_path / "bot.py"
    path.write_text("def x():\n    pass\n")

    calls = {"count": 0}

    def fetch_stub(limit=100, start=None, end=None):
        calls["count"] += 1
        if calls["count"] <= 2:
            return [{"cpu": 1.0, "memory": 1.0}]
        return [{"cpu": 10.0, "memory": 10.0}]

    monkeypatch.setattr(mdb, "fetch", fetch_stub)

    patch_id, reverted, _ = engine.apply_patch(path, "test", threshold=0.1)
    assert patch_id is not None
    assert reverted
    assert "auto_x" not in path.read_text()


def test_apply_patch_verifier_failure(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")

    class DummyVerifier:
        def verify(self, path: Path) -> bool:  # pragma: no cover - simple stub
            return False

    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
        formal_verifier=DummyVerifier(),
    )

    called = {}

    def ci_stub(p=None):
        called["path"] = p
        return True

    monkeypatch.setattr(engine, "_run_ci", ci_stub)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_y():\n pass\n")

    path = tmp_path / "bot.py"
    path.write_text("def y():\n    pass\n")

    patch_id, reverted, _ = engine.apply_patch(path, "test")
    assert patch_id is not None
    assert reverted
    assert "auto_y" not in path.read_text()
    assert patch_db.conn.execute(
        "SELECT reverted FROM patch_history WHERE id=?",
        (patch_id,),
    ).fetchone()[0] == 1
    assert called["path"] == path


def test_sync_git_called_on_success(tmp_path, monkeypatch):
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, patch_db=patch_db)

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_z():\n    pass\n")

    path = tmp_path / "bot.py"
    path.write_text("def z():\n    pass\n")

    calls = []

    def run_stub(cmd, check=True, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr(sce.subprocess, "run", run_stub)

    patch_id, reverted, _ = engine.apply_patch(path, "test")
    assert patch_id is not None
    assert not reverted
    assert ["./sync_git.sh"] in calls


def test_formal_verifier_default(tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem)
    assert engine.formal_verifier is not None


def test_rollback_patch(tmp_path, monkeypatch):
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, patch_db=patch_db)

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    def ci_stub(p=None):
        ci_stub.called = p
        return True

    monkeypatch.setattr(engine, "_run_ci", ci_stub)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_rb():\n    pass\n")

    path = tmp_path / "bot.py"
    path.write_text("def rb():\n    pass\n")

    patch_id, reverted, _ = engine.apply_patch(path, "helper")
    assert patch_id is not None and not reverted
    assert "auto_rb" in path.read_text()

    engine.rollback_patch(str(patch_id))
    assert "auto_rb" not in path.read_text()
    assert ci_stub.called == path


def test_retrieval_context_in_prompt(tmp_path, monkeypatch):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem)

    context = {"errors": [{"id": 1, "snippet": "oops", "note": "test"}]}
    engine.context_builder = types.SimpleNamespace(build_context=lambda m: context)

    class DummyClient:
        def ask(self, messages, memory_manager=None, tags=None, use_memory=True):
            DummyClient.prompt = messages[0]["content"]
            return {"choices": [{"message": {"content": "def auto_test():\n    pass"}}]}

    engine.llm_client = DummyClient()
    monkeypatch.setattr(engine, "suggest_snippets", lambda d, limit=3: [])

    code = engine.generate_helper("test helper")
    assert "### Retrieval context" in DummyClient.prompt
    assert json.dumps(context, indent=2) in DummyClient.prompt
    assert "def auto_test" in code


def test_patch_logger_vector_service_error(tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")

    class BoomLogger:
        def track_contributors(self, *a, **k):
            raise VectorServiceError("nope")

    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        patch_logger=BoomLogger(),
    )

    # Should not raise despite logger failure
    engine._track_contributors("s", [("o", "v")], True)

