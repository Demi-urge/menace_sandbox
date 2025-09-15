from pathlib import Path
import json
import sys
# flake8: noqa
import types
import importlib.util
import dynamic_path_router
import pytest

from llm_interface import LLMResult, Prompt

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [Path(__file__).resolve().parents[1].as_posix()]
sys.modules.setdefault("menace", menace_pkg)

# Lightweight stub for vector_service to avoid heavy imports
vec_mod = types.ModuleType("vector_service")


class _VSError(Exception):
    pass


vec_mod.ContextBuilder = object  # type: ignore[attr-defined]
vec_mod.ErrorResult = object  # type: ignore[attr-defined]
vec_mod.PatchLogger = object  # type: ignore[attr-defined]
vec_mod.VectorServiceError = _VSError
vec_mod.CognitionLayer = object  # type: ignore[attr-defined]
vec_mod.SharedVectorService = object  # type: ignore[attr-defined]


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

builder = types.SimpleNamespace(
    build_context=lambda *a, **k: {},
    refresh_db_weights=lambda *a, **k: None,
    build_prompt=lambda query, intent=None, error_log=None, top_k=5: Prompt(query),
)

sys.modules.setdefault("safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
sys.modules.setdefault("menace.safety_monitor", sys.modules["safety_monitor"])

scm_stub = types.ModuleType("menace.self_coding_manager")
scm_stub.SelfCodingManager = object  # type: ignore[attr-defined]
sys.modules.setdefault("menace.self_coding_manager", scm_stub)

# Stub chunking module to provide expected settings
chunking_stub = types.ModuleType("chunking")
chunking_stub.split_into_chunks = lambda *a, **k: []
chunking_stub.get_chunk_summaries = lambda *a, **k: []
chunking_stub._SETTINGS = types.SimpleNamespace(
    chunk_summary_cache_dir=None, prompt_chunk_cache_dir=None
)
sys.modules.setdefault("chunking", chunking_stub)


from menace.coding_bot_interface import manager_generate_helper  # noqa: E402

def test_refresh_db_weights_failure(tmp_path):
    import menace.self_coding_engine as sce
    import menace.code_database as cd

    bad = types.SimpleNamespace(
        build_context=lambda *a, **k: {},
        refresh_db_weights=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    mem = types.SimpleNamespace()
    with pytest.raises(RuntimeError):
        sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, context_builder=bad)

from vector_service import VectorServiceError

sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object))
cd_stub = types.ModuleType("menace.code_database")
class _CodeDB:
    def __init__(self, *a, **k):
        pass


class _PatchHistoryDB:
    def __init__(self, *a, **k):
        pass


cd_stub.CodeDB = _CodeDB
cd_stub.CodeRecord = object
cd_stub.PatchHistoryDB = _PatchHistoryDB
cd_stub.PatchRecord = object
sys.modules["code_database"] = cd_stub
sys.modules["menace.code_database"] = cd_stub
msl_stub = types.SimpleNamespace(fetch_recent_billing_issues=lambda *a, **k: [])
sys.modules["menace.menace_sanity_layer"] = msl_stub
sys.modules["sandbox_settings"] = types.SimpleNamespace(
    SandboxSettings=lambda: types.SimpleNamespace(
        prompt_chunk_token_threshold=1000, codex_retry_delays=[2, 5, 10]
    )
)
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

    def fetch(self, *a, **k):  # pragma: no cover - simple stub
        return []


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
    def __init__(self, success, stdout="", stderr="", duration=0.0, failure=None, path=None):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.failure = failure
        self.path = path


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
mm_stub.MemoryEntry = object
sys.modules.setdefault("menace_memory_manager", mm_stub)
sys.modules.setdefault("menace.menace_memory_manager", mm_stub)
sys.modules.setdefault(
    "menace.run_autonomous", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)

spec = importlib.util.spec_from_file_location(
    "menace.self_coding_engine",
    dynamic_path_router.path_for_prompt("self_coding_engine.py"),  # path-ignore
)
sce = importlib.util.module_from_spec(spec)
sys.modules["menace.self_coding_engine"] = sce
spec.loader.exec_module(sce)
import menace.code_database as cd  # noqa: E402
import menace.menace_memory_manager as mm  # noqa: E402
import menace.data_bot as db  # noqa: E402


def test_apply_patch_reverts_on_complexity(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    tracker = sce.BaselineTracker()
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
        delta_tracker=tracker,
        context_builder=builder,
    )

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_x():\n    pass\n")

    path = tmp_path / "bot.py"  # path-ignore
    path.write_text("def x():\n    pass\n")

    calls = {"count": 0}

    def fetch_stub(limit=100, start=None, end=None):
        calls["count"] += 1
        if calls["count"] <= 2:
            return [{"cpu": 1.0, "memory": 1.0}]
        return [{"cpu": 10.0, "memory": 10.0}]

    monkeypatch.setattr(mdb, "fetch", fetch_stub)

    tracker.update(
        roi_delta=1.0,
        error_delta=0.0,
        complexity_delta=0.0,
        pred_roi_delta=1.0,
        pred_err_delta=0.0,
    )

    patch_id, reverted, _ = engine.apply_patch(path, "test")
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
        context_builder=builder,
    )

    called = {}

    def ci_stub(p=None):
        called["path"] = p
        return True

    monkeypatch.setattr(engine, "_run_ci", ci_stub)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_y():\n pass\n")

    path = tmp_path / "bot.py"  # path-ignore
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
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, patch_db=patch_db, context_builder=builder)

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

    path = tmp_path / "bot.py"  # path-ignore
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
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, context_builder=builder)
    assert engine.formal_verifier is not None


def test_rollback_patch(tmp_path, monkeypatch):
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, patch_db=patch_db, context_builder=builder)

    class OkVerifier:
        def verify(self, path: Path) -> bool:
            return True

    engine.formal_verifier = OkVerifier()

    def ci_stub(p=None):
        ci_stub.called = p
        return True

    monkeypatch.setattr(engine, "_run_ci", ci_stub)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_rb():\n    pass\n")

    path = tmp_path / "bot.py"  # path-ignore
    path.write_text("def rb():\n    pass\n")

    patch_id, reverted, _ = engine.apply_patch(path, "helper")
    assert patch_id is not None and not reverted
    assert "auto_rb" in path.read_text()

    engine.rollback_patch(str(patch_id))
    assert "auto_rb" not in path.read_text()
    assert ci_stub.called == path


def test_retrieval_context_in_prompt(tmp_path, monkeypatch):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, context_builder=builder)

    context = {"errors": [{"id": 1, "snippet": "oops", "note": "test"}]}
    context_json = json.dumps(context, indent=2)

    class RecordingBuilder:
        def __init__(self):
            self.calls = []

        def build(self, query):  # type: ignore[override]
            self.calls.append(query)
            return context_json

        build_context = build

        def refresh_db_weights(self):
            pass

    engine.context_builder = RecordingBuilder()

    class DummyClient:
        def generate(self, prompt):
            DummyClient.prompt = getattr(prompt, "text", str(prompt))
            return LLMResult(text="def auto_test():\n    pass")

    engine.llm_client = DummyClient()
    monkeypatch.setattr(engine, "suggest_snippets", lambda d, limit=3: [])

    manager = types.SimpleNamespace(engine=engine)
    code = manager_generate_helper(manager, "test helper")
    assert engine.context_builder.calls, "context_builder.build was not invoked"
    assert "### Retrieval context" in DummyClient.prompt
    assert context_json in DummyClient.prompt
    assert "def auto_test" in code


def test_generate_helper_requires_context_builder(tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")
    engine = sce.SelfCodingEngine(cd.CodeDB(tmp_path / "c.db"), mem, context_builder=builder)
    engine.context_builder = None
    manager = types.SimpleNamespace(engine=engine)
    with pytest.raises(RuntimeError):
        manager_generate_helper(manager, "demo task")


def test_patch_logger_vector_service_error(tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")

    class BoomLogger:
        def track_contributors(self, *a, **k):
            raise VectorServiceError("nope")

    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        patch_logger=BoomLogger(),
        context_builder=builder,
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
    class DummyClient2:
        def generate(self, prompt):
            return LLMResult(text="")

    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
        context_builder=builder,
        llm_client=DummyClient2(),
    )
    manager = types.SimpleNamespace(engine=engine)
    code = manager_generate_helper(manager, "demo task")
    assert builder.calls == ["demo task"]
    assert g1.inc_calls == 1
    assert "sentinel_fallback" not in code


def test_call_codex_with_backoff_retries(monkeypatch):
    delays = [1, 2, 3]
    monkeypatch.setattr(sce._settings, "codex_retry_delays", delays)
    sleeps: list[float] = []
    monkeypatch.setattr(sce.time, "sleep", lambda d: sleeps.append(d))

    class FailClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise Exception("boom")

    client = FailClient()
    with pytest.raises(sce.RetryError):
        sce.call_codex_with_backoff(client, sce.Prompt("x"))

    assert sleeps == delays
    assert client.calls == len(delays) + 1


@pytest.mark.skip(reason="outdated after context builder refactor")
def test_codex_fallback_handler_invoked(monkeypatch, tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")

    class DummyClient:
        def generate(self, prompt):
            return LLMResult(text="")

    engine = object.__new__(sce.SelfCodingEngine)
    engine.llm_client = DummyClient()
    engine.suggest_snippets = lambda *a, **k: []
    engine._extract_statements = lambda *a, **k: []
    engine._fetch_retry_trace = lambda *a, **k: ""
    engine.prompt_engine = types.SimpleNamespace(build_prompt=lambda *a, **k: sce.Prompt("code"))
    engine.gpt_memory = types.SimpleNamespace(log_interaction=lambda *a, **k: None, store=lambda *a, **k: None)
    engine.memory_mgr = mem
    engine.knowledge_service = None
    engine.prompt_tone = "neutral"
    engine.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    engine._last_prompt_metadata = {}
    engine._last_prompt = None
    engine._last_retry_trace = None
    engine.simplify_prompt = sce.simplify_prompt

    calls: list[Path] = []

    def handle(prompt, reason, *, context_builder, queue_path=None, **_):
        calls.append(queue_path)
        return LLMResult(text="def good():\n    pass\n")

    monkeypatch.setattr(sce.codex_fallback_handler, "handle", handle)
    monkeypatch.setattr(
        sce,
        "create_context_builder",
        lambda: types.SimpleNamespace(
            refresh_db_weights=lambda *a, **k: None, build=lambda *a, **k: ""
        ),
    )

    qpath = tmp_path / "queue.jsonl"
    monkeypatch.setattr(
        sce,
        "_settings",
        types.SimpleNamespace(codex_retry_delays=[2, 5, 10], codex_retry_queue_path=str(qpath)),
    )
    manager = types.SimpleNamespace(engine=engine)
    code = manager_generate_helper(manager, "demo")
    assert "def good" in code
    assert calls == [qpath]


def test_simplified_prompt_after_failure(monkeypatch, tmp_path):
    mem = mm.MenaceMemoryManager(tmp_path / "m.db")

    calls: list[sce.Prompt] = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise Exception("fail")
        return LLMResult(text="def ok():\n    pass\n")

    client = types.SimpleNamespace(generate=generate)

    def failing_call(client, prompt, *, logger=None, timeout=30.0):
        try:
            return client.generate(prompt)
        except Exception as exc:
            raise sce.RetryError(str(exc))

    monkeypatch.setattr(sce, "call_codex_with_backoff", failing_call)

    engine = object.__new__(sce.SelfCodingEngine)
    engine.llm_client = client
    engine.suggest_snippets = lambda *a, **k: []
    engine._extract_statements = lambda *a, **k: []
    engine._fetch_retry_trace = lambda *a, **k: ""
    engine.prompt_engine = types.SimpleNamespace()
    engine.context_builder = types.SimpleNamespace(
        build_prompt=lambda q, intent=None, error_log=None, top_k=5: sce.Prompt(
            q, system=intent.get("system", "orig") if intent else "orig", examples=intent.get("examples", ["e1", "e2"]) if intent else ["e1", "e2"]
        ),
        build_context=lambda *a, **k: {},
        refresh_db_weights=lambda *a, **k: None,
    )
    engine.gpt_memory = types.SimpleNamespace(log_interaction=lambda *a, **k: None, store=lambda *a, **k: None)
    engine.memory_mgr = mem
    engine.knowledge_service = None
    engine.prompt_tone = "neutral"
    engine.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    engine._last_prompt_metadata = {}
    engine._last_prompt = None
    engine._last_retry_trace = None
    engine.simplify_prompt = sce.simplify_prompt

    manager = types.SimpleNamespace(engine=engine)
    code = manager_generate_helper(manager, "demo")
    assert "def ok" in code
    assert len(calls) == 2
    assert calls[0].system == "orig" and len(calls[0].examples) == 2
    assert calls[1].system == "" and len(calls[1].examples) == 0


def test_generate_helper_requires_manager_token(tmp_path):
    import menace.self_coding_engine as sce

    class DummyManager:
        def __init__(self, engine):
            self.engine = engine

    memory = types.SimpleNamespace()
    engine = sce.SelfCodingEngine(
        cd_stub.CodeDB(tmp_path / "c.db"),
        memory,
        context_builder=builder,
    )

    with pytest.raises(RuntimeError):
        manager_generate_helper(types.SimpleNamespace(engine=engine), "demo")

    mgr_mod = types.ModuleType("menace.self_coding_manager")
    mgr_mod.SelfCodingManager = DummyManager
    sys.modules["menace.self_coding_manager"] = mgr_mod

    from menace.coding_bot_interface import manager_generate_helper

    manager = DummyManager(engine)
    code = manager_generate_helper(manager, "demo")
    assert "def" in code
