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


class _EmbeddableDBMixin:
    def __init__(self, *a, **k):
        pass


vec_mod.EmbeddableDBMixin = _EmbeddableDBMixin  # type: ignore[attr-defined]

class FallbackResult(list):
    def __init__(self, reason, results=None):
        super().__init__(results or [])
        self.reason = reason
vec_mod.FallbackResult = FallbackResult

# minimal decorators module with gauges
dec_mod = types.ModuleType("vector_service.decorators")
class _Gauge:
    def __init__(self):
        self.calls = 0
        self.sets = []
    def labels(self, *args):
        return self
    def inc(self):
        self.calls += 1
    def set(self, value):
        self.sets.append(value)
_dec_call = _Gauge()
_dec_lat = _Gauge()
_dec_size = _Gauge()
dec_mod._CALL_COUNT = _dec_call
dec_mod._LATENCY_GAUGE = _dec_lat
dec_mod._RESULT_SIZE_GAUGE = _dec_size

def log_and_measure(func):
    def wrapper(*a, **k):
        dec_mod._CALL_COUNT.labels(func.__qualname__).inc()
        result = func(*a, **k)
        dec_mod._LATENCY_GAUGE.labels(func.__qualname__).set(0.0)
        try:
            size = len(result)
        except Exception:
            size = 0
        dec_mod._RESULT_SIZE_GAUGE.labels(func.__qualname__).set(size)
        return result
    return wrapper
dec_mod.log_and_measure = log_and_measure
sys.modules.setdefault("vector_service.decorators", dec_mod)
sys.modules.setdefault("vector_service", vec_mod)

from vector_service import VectorServiceError

sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(CodeDB=object))
sys.modules.setdefault(
    "gpt_memory",
    types.SimpleNamespace(
        GPTMemoryManager=object,
        STANDARD_TAGS=[],
        INSIGHT="insight",
        _summarise_text=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace.shared_knowledge_module",
    types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None),
)
sys.modules.setdefault(
    "menace.local_knowledge_module",
    types.SimpleNamespace(LocalKnowledgeModule=object, init_local_knowledge=lambda *a, **k: None),
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
for name in [
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "gpt_knowledge_service",
]:
    sys.modules.setdefault(name, sys.modules[f"menace.{name}"])

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
db_stub = types.ModuleType("data_bot")


class _MetricsDB:
    def __init__(self, *_a, **_k):
        pass


class _DataBot:
    def __init__(self, mdb, patch_db=None):
        self.db = mdb

    def roi(self, *_a, **_k):
        return 0.0

    def complexity_score(self, *_a, **_k):
        return 0.0


db_stub.MetricsDB = _MetricsDB
db_stub.DataBot = _DataBot
sys.modules.setdefault("data_bot", db_stub)
sys.modules.setdefault("menace.data_bot", db_stub)
cg_stub = types.ModuleType("chatgpt_idea_bot")


class _ChatGPTClient:
    def __init__(self, *a, **k):
        pass


cg_stub.ChatGPTClient = _ChatGPTClient
sys.modules.setdefault("chatgpt_idea_bot", cg_stub)
sys.modules.setdefault("menace.chatgpt_idea_bot", cg_stub)
th_stub = types.ModuleType("sandbox_runner.test_harness")


class _THResult:
    def __init__(self, success, stdout="", stderr="", duration=0.0):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration


def _run_tests(*_a, **_k):
    return _THResult(True)


th_stub.run_tests = _run_tests
th_stub.TestHarnessResult = _THResult
sys.modules.setdefault("sandbox_runner.test_harness", th_stub)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)
ws_stub = types.ModuleType("sandbox_runner.workflow_sandbox_runner")


class _WSRunner:
    def run(self, func, test_data=None):
        class _M:
            result = True

        class _Metrics:
            modules = [_M()]

        return _Metrics()


ws_stub.WorkflowSandboxRunner = _WSRunner
sys.modules.setdefault("sandbox_runner.workflow_sandbox_runner", ws_stub)
sys.modules.setdefault("menace.sandbox_runner.workflow_sandbox_runner", ws_stub)
mm_stub = types.ModuleType("menace_memory_manager")


class _MMM:
    def __init__(self, *a, **k):
        pass


mm_stub.MenaceMemoryManager = _MMM
sys.modules.setdefault("menace_memory_manager", mm_stub)
sys.modules.setdefault("menace.menace_memory_manager", mm_stub)
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


def test_vector_service_metrics_and_fallback(tmp_path, monkeypatch):
    import vector_service
    import vector_service.decorators as dec
    from vector_service.decorators import log_and_measure
    import menace.self_coding_engine as sce
    import menace.code_database as cd
    import menace.menace_memory_manager as mm
    import menace.data_bot as db

    class Gauge:
        def __init__(self):
            self.inc_calls = 0
            self.set_calls: list[float] = []
        def labels(self, *args):
            return self
        def inc(self):
            self.inc_calls += 1
        def set(self, value):
            self.set_calls.append(value)

    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    class DummyRetriever:
        @log_and_measure
        def search(self, query, **_):
            return vector_service.FallbackResult("sentinel_fallback", [])

    class DummyBuilder:
        def __init__(self):
            self.calls = []
            self.retriever = DummyRetriever()
        def build(self, query):
            self.calls.append(query)
            return self.retriever.search(query, session_id="s")

    builder = DummyBuilder()

    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
        context_builder=builder,
        llm_client=object(),
    )
    monkeypatch.setattr(sce, "ask_with_memory", lambda *a, **k: {})
    code = engine.generate_helper("demo task")
    assert builder.calls == ["demo task"]
    assert g1.inc_calls == 1
    assert "sentinel_fallback" not in code
