import ast
import logging
from pathlib import Path
from dynamic_path_router import resolve_path

import pytest
import importlib.util
import sys
import types
from menace.coding_bot_interface import manager_generate_helper

cd_stub = types.SimpleNamespace(
    CodeDB=object,
    CodeRecord=object,
    PatchHistoryDB=object,
    PatchRecord=object,
)
sys.modules.setdefault("code_database", cd_stub)
sys.modules.setdefault("menace.code_database", cd_stub)

sys.modules.setdefault(
    "gpt_memory",
    types.SimpleNamespace(
        GPTMemoryManager=object,
        INSIGHT="INSIGHT",
        _summarise_text=lambda text, *a, **k: text,
    ),
)
sys.modules.setdefault(
    "gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object)
)
sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, DBRouter=object, init_db_router=lambda *a, **k: None)
)
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(CognitionLayer=object, SharedVectorService=object, ContextBuilder=object),
)
sys.modules.setdefault(
    "menace.trend_predictor", types.SimpleNamespace(TrendPredictor=object)
)
sys.modules.setdefault("trend_predictor", sys.modules["menace.trend_predictor"])
sys.modules.setdefault(
    "menace.safety_monitor", types.SimpleNamespace(SafetyMonitor=object)
)
sys.modules.setdefault("safety_monitor", sys.modules["menace.safety_monitor"])
sys.modules.setdefault(
    "llm_interface", types.SimpleNamespace(Prompt=str, LLMResult=object, LLMClient=object)
)
sys.modules.setdefault(
    "llm_registry",
    types.SimpleNamespace(create_backend=lambda *a, **k: None, register_backend_from_path=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=object)
)
sys.modules.setdefault(
    "rate_limit", types.SimpleNamespace(estimate_tokens=lambda *a, **k: 0)
)
sys.modules.setdefault(
    "menace.rate_limit", types.SimpleNamespace(estimate_tokens=lambda *a, **k: 0)
)
sys.modules.setdefault(
    "menace.llm_router", types.SimpleNamespace(client_from_settings=lambda *a, **k: None)
)
sys.modules.setdefault("llm_router", sys.modules["menace.llm_router"])
sys.modules.setdefault(
    "shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "shared_knowledge_module", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
sys.modules.setdefault(
    "gpt_knowledge_service", sys.modules["menace.gpt_knowledge_service"]
)
sys.modules.setdefault(
    "governed_retrieval", types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x)
)
sys.modules.setdefault(
    "knowledge_retriever",
    types.SimpleNamespace(
        get_feedback=lambda *a, **k: [],
        get_error_fixes=lambda *a, **k: [],
        recent_feedback=lambda *a, **k: "",
        recent_error_fix=lambda *a, **k: "",
        recent_improvement_path=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.menace_sanity_layer", types.SimpleNamespace(fetch_recent_billing_issues=lambda: "")
)
sys.modules.setdefault("menace_sanity_layer", sys.modules["menace.menace_sanity_layer"])
sys.modules.setdefault(
    "menace.patch_suggestion_db", types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object)
)
sys.modules.setdefault("patch_suggestion_db", sys.modules["menace.patch_suggestion_db"])


class _PromptEngine:
    def __init__(self, *a, **k):
        self.context_builder = k.get("context_builder")
        self.last_metadata = {}

    def build_prompt(self, task, *, context=None, retrieval_context=None, **_):
        parts = []
        if retrieval_context:
            parts.append(retrieval_context)
        if context:
            parts.append(context)
        parts.append(task)
        text = "\n".join(parts)
        return types.SimpleNamespace(text=text)


sys.modules.setdefault(
    "prompt_engine",
    types.SimpleNamespace(PromptEngine=_PromptEngine, _ENCODER=None, diff_within_target_region=lambda *a, **k: False),
)
sys.modules.setdefault("menace.prompt_engine", sys.modules["prompt_engine"])
sys.modules.setdefault(
    "chunking",
    types.SimpleNamespace(
        split_into_chunks=lambda *a, **k: [],
        get_chunk_summaries=lambda *a, **k: [],
        summarize_code=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.failure_retry_utils",
    types.SimpleNamespace(check_similarity_and_warn=lambda *a, **k: None, record_failure=lambda *a, **k: None),
)
sys.modules.setdefault("failure_retry_utils", sys.modules["menace.failure_retry_utils"])
sys.modules.setdefault(
    "menace.failure_fingerprint_store", types.SimpleNamespace(FailureFingerprintStore=object)
)
sys.modules.setdefault("failure_fingerprint_store", sys.modules["menace.failure_fingerprint_store"])
sys.modules.setdefault(
    "menace.failure_fingerprint",
    types.SimpleNamespace(
        FailureFingerprint=object,
        find_similar=lambda *a, **k: [],
        log_fingerprint=lambda *a, **k: None,
    ),
)
sys.modules.setdefault("failure_fingerprint", sys.modules["menace.failure_fingerprint"])
sys.modules.setdefault("error_vectorizer", types.SimpleNamespace(ErrorVectorizer=object))

spec = importlib.util.spec_from_file_location("menace", resolve_path("__init__.py"))
menace_pkg = importlib.util.module_from_spec(spec)
menace_pkg.__path__ = [str(Path().resolve())]
sys.modules.setdefault("menace", menace_pkg)
spec.loader.exec_module(menace_pkg)

spec = importlib.util.spec_from_file_location(
    "menace.self_coding_engine", resolve_path("self_coding_engine.py")
)
sce = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace.self_coding_engine", sce)
sys.modules.setdefault("self_coding_engine", sce)
spec.loader.exec_module(sce)
import menace.coding_bot_interface as mci
mci.MANAGER_CONTEXT = sce.MANAGER_CONTEXT
# ``manager_generate_helper`` captures the module globals at import time, so
# explicitly realign its ``MANAGER_CONTEXT`` reference with the freshly loaded
# engine module for deterministic unit behaviour.
manager_generate_helper.__globals__["MANAGER_CONTEXT"] = sce.MANAGER_CONTEXT
sce._settings.prompt_chunk_token_threshold = 4000
sce._settings.chunk_summary_cache_dir = "."
sce._settings.prompt_success_log_path = "s.log"
sce._settings.prompt_failure_log_path = "f.log"
sce._settings.audit_log_path = "audit.log"
sce._settings.audit_privkey = None
sce._settings.codex_retry_delays = []
sce.time = types.SimpleNamespace(sleep=lambda _: None)
sce.log_prompt_attempt = lambda *a, **k: None


def _build_check_permission():
    src = resolve_path("self_coding_engine.py").read_text()
    tree = ast.parse(src)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfCodingEngine")
    method = next(m for m in class_node.body if isinstance(m, ast.FunctionDef) and m.name == "_check_permission")
    new_class = ast.ClassDef("SelfCodingEngine", [], [], [method], [])
    module = ast.Module([new_class], type_ignores=[])
    module = ast.fix_missing_locations(module)

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("menace", resolve_path("__init__.py"))
    menace = importlib.util.module_from_spec(spec)
    menace.__path__ = [str(Path().resolve())]
    sys.modules.setdefault("menace", menace)
    spec.loader.exec_module(menace)
    from menace.access_control import READ, WRITE, check_permission

    ns = {"READ": READ, "WRITE": WRITE, "check_permission": check_permission}
    exec(compile(module, "<ast>", "exec"), ns)
    return ns["SelfCodingEngine"]


def _integrate_insights(engine, description):
    from menace.log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT

    def recent_feedback(svc):
        return "fb"

    def recent_improvement_path(svc):
        return "path"

    def recent_error_fix(svc):
        return "fix"

    insight_lines = []
    if engine.knowledge_service:
        for label, func in [
            (FEEDBACK, recent_feedback),
            (IMPROVEMENT_PATH, recent_improvement_path),
            (ERROR_FIX, recent_error_fix),
        ]:
            insight = func(engine.knowledge_service)
            if insight:
                insight_lines.append(f"{label} insight: {insight}")
    combined = "\n".join(insight_lines)
    if combined:
        engine.logger.info(
            "patch history context",
            extra={"description": description, "history": combined, "tags": [INSIGHT]},
        )
    return combined


def test_permission_checks():
    Engine = _build_check_permission()
    eng = Engine()
    from menace.access_control import READ, WRITE

    eng.bot_roles = {"bob": READ, "alice": WRITE}
    with pytest.raises(PermissionError):
        eng._check_permission("write", "bob")
    eng._check_permission("write", "alice")


def test_insight_integration(caplog):
    engine = type("E", (), {})()
    engine.knowledge_service = object()
    engine.logger = logging.getLogger("SelfCodingEngine")
    with caplog.at_level(logging.INFO):
        hist = _integrate_insights(engine, "desc")
    assert "fb" in hist and "path" in hist and "fix" in hist
    assert any("patch history context" in r.message for r in caplog.records)


def test_call_codex_with_backoff_retries(monkeypatch):
    delays = [2, 5, 10]
    monkeypatch.setattr(sce._settings, "codex_retry_delays", delays)
    sleeps: list[float] = []

    def fake_retry(func, attempts, delays, logger):
        for _ in range(attempts):
            try:
                func()
            except Exception:
                pass
        sleeps.extend(delays)
        raise sce.RetryError("boom")

    monkeypatch.setattr(sce, "retry_with_backoff", fake_retry)

    class FailClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, *, context_builder=None):
            self.calls += 1
            raise Exception("boom")

    client = FailClient()
    builder = object()
    with pytest.raises(sce.RetryError):
        sce.call_codex_with_backoff(client, sce.Prompt("x"), context_builder=builder)

    assert sleeps == delays
    assert client.calls == len(delays) + 1


def test_context_builder_shared(monkeypatch):
    class DummyBuilder:
        def __init__(self, *_, **kwargs):
            self.roi_tracker = kwargs.get("roi_tracker")
            self.stack_config = types.SimpleNamespace(
                enabled=False, top_k=3, summary_tokens=5, text_max_tokens=10
            )
            self.calls: list[dict] = []

        def refresh_db_weights(self):
            return None

        def build_context(self, query, **__):
            self.calls.append(__)
            if __.get("return_metadata"):
                return f"ctx:{query}", {}
            return f"ctx:{query}"

        def build_prompt(self, query, **_):
            text = f"### Retrieval context\nctx:{query}"
            return types.SimpleNamespace(text=text, user=text, metadata={})

    class DummyLayer:
        def __init__(self, *_, **kwargs):
            self.context_builder = kwargs.get("context_builder")

    monkeypatch.setattr(sce, "ContextBuilder", DummyBuilder)
    monkeypatch.setattr(sce, "CognitionLayer", DummyLayer)

    code_db = types.SimpleNamespace(search=lambda q: [])
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [],
        log_interaction=lambda *a, **k: None,
        store=lambda *a, **k: None,
    )
    class DummyClient:
        def __init__(self):
            self.last_prompt = ""

        def generate(self, prompt, *, context_builder=None):
            self.last_prompt = getattr(prompt, "user", getattr(prompt, "text", str(prompt)))
            return types.SimpleNamespace(text="ok", raw="ok")

    client = DummyClient()
    builder = sce.ContextBuilder()
    engine = sce.SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        gpt_memory=gpt_mem,
        context_builder=builder,
    )

    builder = engine.context_builder
    assert builder is engine.prompt_engine.context_builder
    assert builder is engine.cognition_layer.context_builder

    manager = types.SimpleNamespace(engine=engine)
    manager_generate_helper(manager, "alpha issue", context_builder=builder)
    assert "### Retrieval context" in client.last_prompt
    assert "ctx:alpha issue" in client.last_prompt
    assert any(call.get("return_metadata") for call in builder.calls)


def test_stack_snippets_appended(monkeypatch):
    monkeypatch.setattr(sce._settings, "stack_enabled", True, raising=False)
    class DummyBuilder:
        def __init__(self, *_, **kwargs):
            self.roi_tracker = kwargs.get("roi_tracker")
            self.stack_config = types.SimpleNamespace(
                enabled=True, top_k=3, summary_tokens=50, text_max_tokens=200
            )
            self.calls: list[dict] = []

        def refresh_db_weights(self):
            return None

        def build_context(self, query, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("return_metadata"):
                return f"ctx:{query}", {
                    "stack": [
                        {
                            "repo": "octo/demo",
                            "path": "src/app.py",
                            "desc": "Helpful snippet",
                            "score": 0.9,
                        }
                    ]
                }
            return f"ctx:{query}"

        def build_prompt(self, query, **_):
            text = f"### Retrieval context\nctx:{query}"
            return types.SimpleNamespace(text=text, user=text, metadata={})

    class DummyLayer:
        def __init__(self, *_, **kwargs):
            self.context_builder = kwargs.get("context_builder")

    monkeypatch.setattr(sce, "ContextBuilder", DummyBuilder)
    monkeypatch.setattr(sce, "CognitionLayer", DummyLayer)

    code_db = types.SimpleNamespace(search=lambda q: [])
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [],
        log_interaction=lambda *a, **k: None,
        store=lambda *a, **k: None,
    )

    class DummyClient:
        def __init__(self):
            self.last_prompt = ""

        def generate(self, prompt, *, context_builder=None):
            self.last_prompt = getattr(prompt, "user", getattr(prompt, "text", str(prompt)))
            return types.SimpleNamespace(text="ok", raw="ok")

    client = DummyClient()
    builder = sce.ContextBuilder()
    engine = sce.SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        gpt_memory=gpt_mem,
        context_builder=builder,
    )

    manager = types.SimpleNamespace(engine=engine)
    manager_generate_helper(manager, "alpha issue", context_builder=builder)

    assert sce.STACK_PROMPT_HEADER in client.last_prompt
    assert "Helpful snippet" in client.last_prompt
    assert engine._last_prompt_metadata["stack_context_used"] is True
    assert engine._last_prompt_metadata["stack_snippet_count"] == 1


def test_stack_snippets_deduplicated(monkeypatch):
    monkeypatch.setattr(sce._settings, "stack_enabled", True, raising=False)
    class DummyBuilder:
        def __init__(self, *_, **kwargs):
            self.roi_tracker = kwargs.get("roi_tracker")
            self.stack_config = types.SimpleNamespace(
                enabled=True, top_k=5, summary_tokens=50, text_max_tokens=200
            )

        def refresh_db_weights(self):
            return None

        def build_context(self, query, **kwargs):
            meta = {
                "stack": [
                    {
                        "repo": "octo/demo",
                        "path": "src/app.py",
                        "desc": "Helpful snippet",
                        "score": 0.9,
                    },
                    {
                        "repo": "octo/demo",
                        "path": "src/app.py",
                        "desc": "Helpful snippet",
                        "score": 0.8,
                    },
                ]
            }
            return (f"ctx:{query}", meta) if kwargs.get("return_metadata") else f"ctx:{query}"

        def build_prompt(self, query, **_):
            text = f"### Retrieval context\nctx:{query}"
            return types.SimpleNamespace(text=text, user=text, metadata={})

    class DummyLayer:
        def __init__(self, *_, **kwargs):
            self.context_builder = kwargs.get("context_builder")

    monkeypatch.setattr(sce, "ContextBuilder", DummyBuilder)
    monkeypatch.setattr(sce, "CognitionLayer", DummyLayer)

    class DummyClient:
        def __init__(self):
            self.last_prompt = ""

        def generate(self, prompt, **_):
            self.last_prompt = getattr(prompt, "user", getattr(prompt, "text", str(prompt)))
            return types.SimpleNamespace(text="ok", raw="ok")

    client = DummyClient()

    code_db = types.SimpleNamespace(search=lambda q: [])
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [],
        log_interaction=lambda *a, **k: None,
        store=lambda *a, **k: None,
    )

    builder = sce.ContextBuilder()
    engine = sce.SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        gpt_memory=gpt_mem,
        context_builder=builder,
    )

    manager = types.SimpleNamespace(engine=engine)
    manager_generate_helper(manager, "alpha issue", context_builder=builder)

    meta = engine._last_prompt_metadata
    assert meta["stack_snippet_count"] == 1
    assert meta["stack_context_used"] is True


def test_stack_snippets_disabled_override(monkeypatch):
    monkeypatch.setattr(sce._settings, "stack_enabled", False, raising=False)
    class DummyBuilder:
        def __init__(self, *_, **kwargs):
            self.roi_tracker = kwargs.get("roi_tracker")
            self.stack_config = types.SimpleNamespace(
                enabled=True, top_k=2, summary_tokens=50, text_max_tokens=200
            )

        def refresh_db_weights(self):
            return None

        def build_context(self, query, **kwargs):
            if kwargs.get("return_metadata"):
                return f"ctx:{query}", {
                    "stack": [
                        {
                            "repo": "octo/demo",
                            "path": "src/app.py",
                            "desc": "Helpful snippet",
                        }
                    ]
                }
            return f"ctx:{query}"

        def build_prompt(self, query, **_):
            text = f"### Retrieval context\nctx:{query}"
            return types.SimpleNamespace(text=text, user=text, metadata={})

    class DummyLayer:
        def __init__(self, *_, **kwargs):
            self.context_builder = kwargs.get("context_builder")

    monkeypatch.setattr(sce, "ContextBuilder", DummyBuilder)
    monkeypatch.setattr(sce, "CognitionLayer", DummyLayer)

    class DummyClient:
        def __init__(self):
            self.last_prompt = ""

        def generate(self, prompt, **_):
            self.last_prompt = getattr(prompt, "user", getattr(prompt, "text", str(prompt)))
            return types.SimpleNamespace(text="ok", raw="ok")

    client = DummyClient()

    code_db = types.SimpleNamespace(search=lambda q: [])
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [],
        log_interaction=lambda *a, **k: None,
        store=lambda *a, **k: None,
    )

    builder = sce.ContextBuilder()
    engine = sce.SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        gpt_memory=gpt_mem,
        context_builder=builder,
        stack_assist=False,
    )

    manager = types.SimpleNamespace(engine=engine)
    manager_generate_helper(manager, "alpha issue", context_builder=builder)

    assert sce.STACK_PROMPT_HEADER not in client.last_prompt
    meta = engine._last_prompt_metadata
    assert meta["stack_context_used"] is False
    assert meta["stack_snippet_count"] == 0

