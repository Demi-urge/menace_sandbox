# flake8: noqa
import json
import os
import importlib.util
import importlib
import sys
import types
from types import ModuleType
from pathlib import Path
import logging
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
stub = ModuleType("db_router")
stub.DBRouter = object
stub.GLOBAL_ROUTER = None
stub.init_db_router = lambda *a, **k: None
stub.LOCAL_TABLES = {}
stub.SHARED_TABLES = {}
stub.queue_insert = lambda *a, **k: None
sys.modules.setdefault("db_router", stub)
sys.modules.setdefault("menace.db_router", stub)
vec_stub = ModuleType("vector_service")
class _CB:
    def __init__(self, *a, **k):
        pass
    def build(self, *_a, **_k):
        return ""

    def build_prompt(self, query, **_k):
        return types.SimpleNamespace(user=query, examples=[], metadata={})
    def refresh_db_weights(self):
        pass
vec_stub.ContextBuilder = _CB
class _FallbackResult:
    def __init__(self, *a, **k):
        pass
vec_stub.FallbackResult = _FallbackResult
vec_stub.ErrorResult = type("ErrorResult", (), {})
vec_stub.EmbeddableDBMixin = object
vec_stub.SharedVectorService = object
sys.modules.setdefault("vector_service", vec_stub)
sys.modules.setdefault("vector_service.context_builder", vec_stub)
vec_dec = ModuleType("vector_service.decorators")
def _log_and_measure(fn):
    def wrapper(*a, **k):
        if getattr(vec_dec._CALL_COUNT, "inc", None):
            vec_dec._CALL_COUNT.inc()
        return fn(*a, **k)
    return wrapper
vec_dec.log_and_measure = _log_and_measure
vec_dec._CALL_COUNT = None
vec_dec._LATENCY_GAUGE = None
vec_dec._RESULT_SIZE_GAUGE = None
sys.modules.setdefault("vector_service.decorators", vec_dec)
sc_stub = ModuleType("snippet_compressor")
sc_stub.compress_snippets = lambda meta, **k: meta
sys.modules.setdefault("snippet_compressor", sc_stub)

code_db_stub = ModuleType("code_database")
code_db_stub.CodeDB = object
sys.modules["code_database"] = code_db_stub
sys.modules["menace.code_database"] = code_db_stub

scm_stub = ModuleType("self_coding_manager")
class _SCM:
    def __init__(self, *a, **k):
        pass
    def run_patch(self, path, prompt):
        path.write_text("")
scm_stub.SelfCodingManager = _SCM
sys.modules.setdefault("self_coding_manager", scm_stub)
sys.modules.setdefault("menace.self_coding_manager", scm_stub)

sys.modules.setdefault("error_vectorizer", types.SimpleNamespace(ErrorVectorizer=object))
sys.modules.setdefault("failure_fingerprint", types.SimpleNamespace(FailureFingerprint=object))
sys.modules.setdefault(
    "failure_fingerprint_store", types.SimpleNamespace(FailureFingerprintStore=object)
)
sys.modules.setdefault(
    "vector_utils", types.SimpleNamespace(cosine_similarity=lambda *a, **k: 0.0)
)

mm_stub = ModuleType("menace_memory_manager")
class _MM:
    pass
mm_stub.MenaceMemoryManager = _MM
sys.modules["menace_memory_manager"] = mm_stub
sys.modules["menace.menace_memory_manager"] = mm_stub

sce_stub = ModuleType("self_coding_engine")


class _DummyEngine:
    def __init__(self, *a, **k):
        self.calls: list[str] = []

    def generate_helper(self, desc: str) -> str:
        self.calls.append(desc)
        return "def run():\n    return None\n"


def _engine() -> _DummyEngine:
    return _DummyEngine()


sce_stub.SelfCodingEngine = _DummyEngine
sys.modules.setdefault("self_coding_engine", sce_stub)
sys.modules.setdefault("menace.self_coding_engine", sce_stub)
pkg_path = os.path.join(os.path.dirname(__file__), "..")
pkg_spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(pkg_path, "__init__.py"), submodule_search_locations=[pkg_path]  # path-ignore
)
menace_pkg = importlib.util.module_from_spec(pkg_spec)
sys.modules["menace"] = menace_pkg
pkg_spec.loader.exec_module(menace_pkg)
bdb = importlib.import_module("menace.bot_development_bot")
cfg_mod = importlib.import_module("menace.bot_dev_config")
bdb.BotDevelopmentBot.lint_code = lambda self, path: None


def _ctx_builder():
    return vec_stub.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")


def test_refresh_db_weights_failure(tmp_path, monkeypatch):
    def bad(builder):
        raise RuntimeError("boom")

    monkeypatch.setattr(bdb, "ensure_fresh_weights", bad)
    with pytest.raises(RuntimeError):
        bdb.BotDevelopmentBot(
            repo_base=tmp_path, context_builder=_ctx_builder(), engine=_engine()
        )


def _spec_dict():
    return {
        "name": "sample_bot",
        "purpose": "demo",
        "functions": ["run"],
        "language": "python",
        "dependencies": ["requests"],
        "level": "L1",
        "description": "Sample bot",
        "function_docs": {"run": "Run the bot"},
    }


def _json():
    return json.dumps(_spec_dict())


def _yaml():
    try:
        import yaml
    except Exception:
        return _json()
    return yaml.safe_dump(_spec_dict())


def test_parse_plan_json():
    bot = bdb.BotDevelopmentBot(
        repo_base="tmp", context_builder=_ctx_builder(), engine=_engine()
    )
    specs = bot.parse_plan(_json())
    assert specs[0].name == "sample_bot"
    assert specs[0].level == "L1"
    assert specs[0].description == "Sample bot"
    assert specs[0].function_docs.get("run") == "Run the bot"


def test_parse_plan_yaml():
    bot = bdb.BotDevelopmentBot(
        repo_base="tmp", context_builder=_ctx_builder(), engine=_engine()
    )
    specs = bot.parse_plan(_yaml())
    assert specs[0].name == "sample_bot"
    assert specs[0].level == "L1"
    assert specs[0].description == "Sample bot"
    assert specs[0].function_docs.get("run") == "Run the bot"


def test_build_from_plan(tmp_path):
    cfg = cfg_mod.BotDevConfig()
    engine = _engine()
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path,
        config=cfg,
        context_builder=_ctx_builder(),
        engine=engine,
    )
    files = bot.build_from_plan(_json())
    assert engine.calls, "generate_helper was not called"
    assert (tmp_path / "sample_bot" / "sample_bot.py") in files  # path-ignore
    assert (tmp_path / "sample_bot" / "sample_bot.py").exists()  # path-ignore
    req = tmp_path / "sample_bot" / "requirements.txt"
    assert req.exists()
    test_file = tmp_path / "sample_bot" / "tests" / "test_run.py"  # path-ignore
    assert test_file.exists()


def test_config_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_DEV_REPO_BASE", str(tmp_path))
    cfg = cfg_mod.BotDevConfig()
    bot = bdb.BotDevelopmentBot(
        config=cfg, context_builder=_ctx_builder(), engine=_engine()
    )
    assert bot.repo_base == tmp_path


def test_build_prompt_with_docs(tmp_path):
    cfg = cfg_mod.BotDevConfig()
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path,
        config=cfg,
        context_builder=_ctx_builder(),
        engine=_engine(),
    )
    spec = bdb.BotSpec(
        name="doc_bot",
        purpose="demo",
        description="Module desc",
        io_format="json",
        level="L2",
        functions=["run_task", "helper"],
        function_docs={"run_task": "Run tasks", "helper": "Assist"},
    )
    prompt = bot._build_prompt(spec, context_builder=bot.context_builder)
    assert "INSTRUCTION MODE" in prompt.user
    assert "Module desc" in prompt.user
    assert "Level: L2" in prompt.user
    assert "IO Format: json" in prompt.user
    assert "- run_task: Run tasks" in prompt.user
    assert "- helper: Assist" in prompt.user


def test_prompt_includes_standards(tmp_path):
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=_ctx_builder(), engine=_engine()
    )
    spec = bdb.BotSpec(name="std_bot", purpose="demo")
    prompt = bot._build_prompt(spec, context_builder=bot.context_builder)
    assert "INSTRUCTION MODE" in prompt.user
    assert "Coding Standards:" in prompt.user
    assert "PEP8" in prompt.user
    assert "Repository Layout:" in prompt.user
    assert "Metadata:" in prompt.user
    assert "meta.yaml" in prompt.user
    assert "Version Control:" in prompt.user
    assert "Testing:" in prompt.user
    assert "setup_tests.sh" in prompt.user


def test_prompt_includes_function_guidance(tmp_path):
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=_ctx_builder(), engine=_engine()
    )
    spec = bdb.BotSpec(name="guide_bot", purpose="demo", functions=["click_target"])
    prompt = bot._build_prompt(spec, context_builder=bot.context_builder)
    assert "Function Guidance:" in prompt.user
    assert "click_target:" in prompt.user


def test_prompt_routes_through_context_builder(tmp_path):
    class DummyBuilder(bdb.ContextBuilder):
        def __init__(self):
            self.calls: list[str] = []

        def build_prompt(self, intent, **_):  # type: ignore[override]
            self.calls.append(intent)
            examples = ["dup", "dup", "extra"]
            deduped: list[str] = []
            for e in examples:
                if e not in deduped:
                    deduped.append(e)
            return bdb.Prompt(user=intent, examples=deduped)

    builder = DummyBuilder()
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=builder, engine=_engine()
    )
    spec = bdb.BotSpec(name="ctx_bot", purpose="demo", description="demo")
    prompt = bot._build_prompt(spec, context_builder=builder)
    assert builder.calls, "context_builder.build_prompt was not invoked"
    assert prompt.examples == ["dup", "extra"]


def test_engine_failure_fallback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(bdb, "Repo", None)

    dev = bdb.BotDevelopmentBot(
        repo_base=tmp_path,
        context_builder=_ctx_builder(),
        engine=_engine(),
    )
    dev.engine_retry = bdb.RetryStrategy(attempts=3, delay=0)

    calls = {"n": 0}

    def boom(_: str) -> str:
        calls["n"] += 1
        raise RuntimeError("bad")

    monkeypatch.setattr(dev.engine, "generate_helper", boom)

    escalated: list[str] = []
    dev._escalate = lambda msg, level="error": escalated.append(msg)

    spec = bdb.BotSpec(name="fallback_bot", purpose="demo", functions=["run"])
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        dev.build_bot(spec, context_builder=dev.context_builder)
    assert calls["n"] == dev.engine_retry.attempts
    assert any("engine request failed" in m for m in escalated)
    assert any("engine request failed" in e for e in dev.errors)


def test_build_from_plan_honours_concurrency(tmp_path, monkeypatch):
    import concurrent.futures as cf

    calls: dict[str, int] = {}

    class DummyExec:
        def __init__(self, max_workers: int) -> None:
            calls["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def map(self, func, iterable):
            return [func(x) for x in iterable]

    monkeypatch.setattr(cf, "ThreadPoolExecutor", DummyExec)
    monkeypatch.setattr(bdb, "Repo", None)

    cfg = cfg_mod.BotDevConfig(concurrency_workers=2)
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path,
        config=cfg,
        context_builder=_ctx_builder(),
        engine=_engine(),
    )
    spec2 = _spec_dict()
    spec2["name"] = "sample_bot2"
    plan = json.dumps([_spec_dict(), spec2])
    paths = bot.build_from_plan(plan)
    assert (tmp_path / "sample_bot" / "sample_bot.py") in paths  # path-ignore
    assert calls.get("workers") == 2


def test_vector_service_metrics_and_fallback(monkeypatch, tmp_path):
    from vector_service import FallbackResult
    import vector_service.decorators as dec
    from vector_service.decorators import log_and_measure

    class Gauge:
        def __init__(self):
            self.inc_calls = 0
            self.set_calls: list[float] = []
            self.labels_args: list[tuple] = []
        def labels(self, *args):
            self.labels_args.append(args)
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
            return FallbackResult("sentinel_fallback", [])

    class DummyBuilder(_CB):
        def __init__(self):
            self.calls = []
            self.retriever = DummyRetriever()
        def build_prompt(self, query, **_):  # type: ignore[override]
            self.calls.append(query)
            self.retriever.search(query, session_id="s")
            return bdb.Prompt(user=query, examples=[])

    builder = DummyBuilder()
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=builder, engine=_engine()
    )
    spec = bdb.BotSpec(name="demo", purpose="demo", description="demo")
    prompt = bot._build_prompt(spec, context_builder=bot.context_builder)
    assert builder.calls == ["demo"]
    assert g1.inc_calls == 1
    assert "sentinel_fallback" not in prompt.user
    assert all("sentinel_fallback" not in ex for ex in prompt.examples)


def test_build_from_plan_passes_context_builder(tmp_path):
    class CaptureBot(bdb.BotDevelopmentBot):
        def __init__(self, repo_base: Path) -> None:  # type: ignore[override]
            super().__init__(
                repo_base=repo_base,
                context_builder=_ctx_builder(),
                engine=_engine(),
            )
            self.received = None

        def build_bot(
            self,
            spec: bdb.BotSpec,
            *,
            model_id=None,
            **kwargs,
        ) -> Path:  # type: ignore[override]
            self.received = self.context_builder
            repo_dir = self.create_env(spec)
            file_path = repo_dir / f"{spec.name}.py"  # path-ignore
            file_path.write_text("pass")
            self._write_meta(repo_dir, spec)
            return file_path

    bot = CaptureBot(repo_base=tmp_path)
    plan = json.dumps([
        {
            "name": "demo",
            "purpose": "demo",
            "functions": ["run"],
            "language": "python",
            "dependencies": [],
            "capabilities": [],
            "level": "",
            "io": "",
        }
    ])
    bot.build_from_plan(plan)
    assert bot.received is bot.context_builder


def test_prompt_context_compression(tmp_path, monkeypatch):
    class SentinelBuilder(bdb.ContextBuilder):
        def __init__(self, **dbs):
            self.dbs = dbs

        def build_prompt(self, intent, **_):  # type: ignore[override]
            raw = "RAW-" + ",".join(sorted(self.dbs.values()))
            comp = bdb.compress_snippets({"snippet": raw}).get("snippet", raw)
            return bdb.Prompt(user=intent, examples=[comp])

        def refresh_db_weights(self):
            pass

    def fake_compress(meta, **_):
        txt = meta.get("snippet", "")
        return {"snippet": txt.replace("RAW-", "COMPRESSED-")}

    monkeypatch.setattr(bdb, "compress_snippets", fake_compress)

    builder = SentinelBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=builder, engine=_engine()
    )
    spec = bdb.BotSpec(name="sentinel", purpose="demo")
    prompt = bot._build_prompt(spec, context_builder=builder)
    assert any(
        "COMPRESSED-bots.db,code.db,errors.db,workflows.db" in ex
        for ex in prompt.examples
    )
    assert all("RAW-bots.db" not in ex for ex in prompt.examples)

    with pytest.raises(ValueError):
        bdb.BotDevelopmentBot(
            repo_base=tmp_path,
            context_builder=None,
            engine=_engine(),
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "secret",
    [
        "api_key=AAAAAAAAAAAAAAAA",
        "password=abcdefg",
        "Bearer A1B2C3D4E5",
        "AKIA0123456789ABCDEF",
    ],
)
def test_call_codex_api_redacts_secrets(secret, caplog, tmp_path, monkeypatch):
    bot = bdb.BotDevelopmentBot(
        repo_base=tmp_path, context_builder=_ctx_builder(), engine=_engine()
    )
    bot.manager = types.SimpleNamespace(engine=_engine())
    monkeypatch.setattr(bdb, "manager_generate_helper", lambda _mgr, _d: "")
    messages = [{"role": "user", "content": f"some {secret} here"}]
    with caplog.at_level(logging.INFO):
        bot._call_codex_api(messages)
    logs = "\n".join(
        r.getMessage() for r in caplog.records if "generate_helper prompt" in r.getMessage()
    )
    assert "[REDACTED]" in logs
    assert secret not in logs
