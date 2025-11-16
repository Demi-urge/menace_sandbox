"""Tests for the enhancement bot context injection."""

import sys
import types
from pathlib import Path
import importlib

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub heavy dependencies so the module imports without optional components
ceb_stub = types.SimpleNamespace(
    EnhancementDB=object,
    EnhancementHistory=object,
    Enhancement=object,
)
sys.modules.setdefault("menace_sandbox.chatgpt_enhancement_bot", ceb_stub)
sys.modules.setdefault("chatgpt_enhancement_bot", ceb_stub)

code_db_stub = types.ModuleType("menace_sandbox.code_database")
code_db_stub.CodeDB = type("CodeDB", (), {"__init__": lambda self, *a, **k: None})
sys.modules["menace_sandbox.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub

# Avoid pulling in heavy transformer stack during imports
transformers_stub = types.SimpleNamespace(AutoTokenizer=None)
sys.modules.setdefault("transformers", transformers_stub)

# Ensure snippet_compressor resolves relative imports correctly
sc_mod = importlib.import_module("menace_sandbox.snippet_compressor")
sys.modules.setdefault("snippet_compressor", sc_mod)

# Minimal dynamic_path_router stub to avoid sandbox bootstrap dependencies
dpr_stub = types.ModuleType("dynamic_path_router")
dpr_stub.resolve_path = lambda name, *_a, **_k: Path(name)
dpr_stub.path_for_prompt = lambda name, *_a, **_k: Path(name)
dpr_stub.repo_root = lambda *_a, **_k: Path(".")
sys.modules["dynamic_path_router"] = dpr_stub
sys.modules["menace_sandbox.dynamic_path_router"] = dpr_stub

# Minimal self_improvement package to satisfy data_bot imports
self_improvement_pkg = types.ModuleType("menace_sandbox.self_improvement")
self_improvement_pkg.__path__ = []  # type: ignore[attr-defined]
baseline_stub = types.ModuleType("menace_sandbox.self_improvement.baseline_tracker")
baseline_stub.BaselineTracker = object
sys.modules["menace_sandbox.self_improvement"] = self_improvement_pkg
sys.modules["menace_sandbox.self_improvement.baseline_tracker"] = baseline_stub

# Lightweight stubs for self-coding infrastructure
bot_registry_stub = types.ModuleType("menace_sandbox.bot_registry")
bot_registry_stub.BotRegistry = type(
    "BotRegistry",
    (),
    {
        "update_bot": lambda *_a, **_k: None,
        "register_bot": lambda *_a, **_k: None,
    },
)
sys.modules["menace_sandbox.bot_registry"] = bot_registry_stub

data_bot_stub = types.ModuleType("menace_sandbox.data_bot")
data_bot_stub.DataBot = type("DataBot", (), {"__init__": lambda self, *a, **k: None})
data_bot_stub.persist_sc_thresholds = lambda *a, **_k: None
sys.modules["menace_sandbox.data_bot"] = data_bot_stub

self_coding_manager_stub = types.ModuleType("menace_sandbox.self_coding_manager")

class _StubManager:
    def __init__(self) -> None:
        self.bot_registry = bot_registry_stub.BotRegistry()
        self.data_bot = data_bot_stub.DataBot()
        self.evolution_orchestrator = object()


self_coding_manager_stub.SelfCodingManager = _StubManager
self_coding_manager_stub.internalize_coding_bot = lambda *a, **_k: _StubManager()
sys.modules["menace_sandbox.self_coding_manager"] = self_coding_manager_stub

self_coding_engine_stub = types.ModuleType("menace_sandbox.self_coding_engine")
self_coding_engine_stub.SelfCodingEngine = type("SelfCodingEngine", (), {"__init__": lambda self, *a, **k: None})
self_coding_engine_stub.MANAGER_CONTEXT = None
sys.modules["menace_sandbox.self_coding_engine"] = self_coding_engine_stub

model_pipeline_stub = types.ModuleType("menace_sandbox.model_automation_pipeline")
model_pipeline_stub.ModelAutomationPipeline = type("ModelAutomationPipeline", (), {"__init__": lambda self, *a, **k: None})
sys.modules["menace_sandbox.model_automation_pipeline"] = model_pipeline_stub

threshold_service_stub = types.ModuleType("menace_sandbox.threshold_service")
threshold_service_stub.ThresholdService = type("ThresholdService", (), {"__init__": lambda self, *a, **k: None})
threshold_service_stub.threshold_service = threshold_service_stub.ThresholdService()
sys.modules["menace_sandbox.threshold_service"] = threshold_service_stub

gpt_memory_stub = types.ModuleType("menace_sandbox.gpt_memory")
gpt_memory_stub.GPTMemoryManager = type("GPTMemoryManager", (), {"__init__": lambda self, *a, **k: None})
sys.modules["menace_sandbox.gpt_memory"] = gpt_memory_stub

self_coding_thresholds_stub = types.ModuleType("menace_sandbox.self_coding_thresholds")
self_coding_thresholds_stub.get_thresholds = lambda *_a, **_k: types.SimpleNamespace(
    roi_drop=0.1,
    error_increase=0.1,
    test_failure_increase=0.1,
)
self_coding_thresholds_stub.update_thresholds = lambda *a, **_k: None
self_coding_thresholds_stub._load_config = lambda *a, **_k: {}
sys.modules["menace_sandbox.self_coding_thresholds"] = self_coding_thresholds_stub

shared_orchestrator_stub = types.ModuleType("menace_sandbox.shared_evolution_orchestrator")
shared_orchestrator_stub.get_orchestrator = lambda *a, **_k: None
sys.modules["menace_sandbox.shared_evolution_orchestrator"] = shared_orchestrator_stub

coding_interface_stub = types.ModuleType("menace_sandbox.coding_bot_interface")

def _identity_decorator(*_a, **_k):
    def _wrap(cls):
        return cls

    return _wrap


coding_interface_stub.self_coding_managed = _identity_decorator
sys.modules["menace_sandbox.coding_bot_interface"] = coding_interface_stub

# Provide minimal vector_service stub
vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.__path__ = []  # type: ignore[attr-defined]
context_builder_mod = types.ModuleType("vector_service.context_builder")
context_builder_mod.ContextBuilder = object
context_builder_mod.record_failed_tags = lambda *a, **_k: None
context_builder_mod.load_failed_tags = lambda *a, **_k: set()
vector_service_pkg.context_builder = context_builder_mod
embedding_stub = types.SimpleNamespace(
    EmbeddingBackfill=types.SimpleNamespace(watch_events=lambda *a, **k: None)
)
vector_service_pkg.embedding_backfill = embedding_stub
vector_service_pkg.EmbeddingBackfill = embedding_stub.EmbeddingBackfill
vector_service_pkg.EmbeddableDBMixin = object
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = context_builder_mod
sys.modules["vector_service.embedding_backfill"] = embedding_stub
sys.modules.setdefault("vector_service.embed_utils", types.ModuleType("vector_service.embed_utils"))

# Micro model stubs
diff_stub = types.SimpleNamespace(summarize_diff=lambda *a, **k: "")
micro_pkg = types.ModuleType("menace_sandbox.micro_models")
sys.modules.setdefault("menace_sandbox.micro_models", micro_pkg)
sys.modules.setdefault("menace_sandbox.micro_models.diff_summarizer", diff_stub)

from menace_sandbox.enhancement_bot import EnhancementBot  # noqa: E402
from llm_interface import LLMClient, LLMResult, Prompt  # noqa: E402


def test_codex_summarize_injects_context():
    calls: dict[str, str | int] = {}

    prompts: dict[str, object] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            if query == "diff":
                calls["desc"] = query
                calls["top_k"] = top_k
                prompt = Prompt(query, examples=["CTX"], metadata={"vector_confidences": [1.0]})
                prompt.origin = "context_builder"
                return prompt
            meta = intent or intent_metadata or {}
            prompts["built"] = Prompt(
                query,
                metadata={"intent": meta, "vector_confidences": [1.0]},
            )
            prompts["built"].origin = "context_builder"
            return prompts["built"]

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.prompt = None
            self.ctx = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            self.prompt = prompt
            self.ctx = context_builder
            return LLMResult(text="summary")

    builder = DummyBuilder()
    llm = DummyLLM()
    bot = EnhancementBot(context_builder=builder, llm_client=llm)
    res = bot._codex_summarize(
        "before",
        "after",
        hint="diff",
        confidence=1.0,
        context_builder=builder,
    )
    assert res == "summary"
    assert calls["desc"] == "diff"
    assert calls["top_k"] == 5
    assert llm.prompt is prompts["built"]
    meta = prompts["built"].metadata["intent"]
    from snippet_compressor import compress_snippets

    expected_ctx = compress_snippets({"snippet": "CTX"})["snippet"]
    assert meta["retrieved_context"] == expected_ctx
    assert meta["refactor_summary"] == "diff"
    assert "before_hash" in meta and "after_hash" in meta
    assert llm.ctx is builder


def test_codex_summarize_compresses_context():
    long_ctx = "X" * 1000
    from snippet_compressor import compress_snippets
    expected = compress_snippets({"snippet": long_ctx})["snippet"]

    prompts: dict[str, Prompt] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            if query == "diff":
                prompt = Prompt(
                    query,
                    examples=[long_ctx],
                    metadata={"vector_confidences": [1.0]},
                )
                prompt.origin = "context_builder"
                return prompt
            meta = intent or intent_metadata or {}
            prompts["built"] = Prompt(
                query,
                metadata={"intent": meta, "vector_confidences": [1.0]},
            )
            prompts["built"].origin = "context_builder"
            return prompts["built"]

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.prompt = None
            self.ctx = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            self.prompt = prompt
            self.ctx = context_builder
            return LLMResult(text="summary")

    builder = DummyBuilder()
    bot = EnhancementBot(context_builder=builder, llm_client=DummyLLM())
    bot._codex_summarize(
        "before",
        "after",
        hint="diff",
        confidence=1.0,
        context_builder=builder,
    )
    assert bot.llm_client and bot.llm_client.prompt  # type: ignore[attr-defined]
    meta = prompts["built"].metadata["intent"]
    assert meta["retrieved_context"] == expected
    assert meta["retrieved_context"] != long_ctx


def test_codex_summarize_builds_prompt():
    prompts: dict[str, Prompt] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            prompts["built"] = Prompt(
                query,
                metadata={
                    "intent": intent or intent_metadata or {},
                    "vector_confidences": [1.0],
                },
            )
            prompts["built"].origin = "context_builder"
            return prompts["built"]

    called: dict[str, object] = {}

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            called["prompt"] = prompt
            return LLMResult(text="ok")

    builder = DummyBuilder()
    bot = EnhancementBot(context_builder=builder, llm_client=DummyLLM())
    bot._codex_summarize("before", "after", hint="diff", context_builder=builder)
    assert "built" in prompts, "context_builder.build_prompt was not invoked"
    assert called["prompt"] is prompts["built"], "llm_client.generate did not receive built prompt"


def test_requires_context_builder():
    with pytest.raises(ValueError):
        EnhancementBot(context_builder=None)  # type: ignore[arg-type]
