from unittest.mock import MagicMock
import sys
import types
import time
import pytest

# Stub modules with heavy dependencies before importing llm_interface
rate_limit_stub = sys.modules.setdefault("rate_limit", types.ModuleType("rate_limit"))
rate_limit_stub.estimate_tokens = lambda text: len(str(text))
llm_pricing_stub = sys.modules.setdefault("llm_pricing", types.ModuleType("llm_pricing"))
llm_config_stub = sys.modules.setdefault("llm_config", types.ModuleType("llm_config"))

from llm_interface import LLMResult  # noqa: E402
from prompt_types import Prompt  # noqa: E402
from menace.coding_bot_interface import manager_generate_helper

# Stub heavy dependencies required by self_coding_engine at import time
code_db_stub = sys.modules.setdefault("code_database", types.ModuleType("code_database"))
code_db_stub.CodeDB = object
code_db_stub.CodeRecord = object
code_db_stub.PatchHistoryDB = object
code_db_stub.PatchRecord = object
sys.modules.setdefault("menace.code_database", code_db_stub)

ue_stub = sys.modules.setdefault("unified_event_bus", types.ModuleType("unified_event_bus"))
ue_stub.UnifiedEventBus = object
sys.modules.setdefault("menace.unified_event_bus", ue_stub)

sgm_stub = sys.modules.setdefault("shared_gpt_memory", types.ModuleType("shared_gpt_memory"))
sgm_stub.GPT_MEMORY_MANAGER = object
sys.modules.setdefault("menace.shared_gpt_memory", sgm_stub)

gpt_mem_stub = sys.modules.setdefault("gpt_memory", types.ModuleType("gpt_memory"))
gpt_mem_stub.GPTMemoryManager = object
gpt_mem_stub.INSIGHT = "INSIGHT"
gpt_mem_stub._summarise_text = lambda text, *a, **k: text
sys.modules.setdefault("menace.gpt_memory", gpt_mem_stub)

vector_stub = sys.modules.setdefault("vector_service", types.ModuleType("vector_service"))
vector_stub.SharedVectorService = object
vector_stub.CognitionLayer = object
vector_stub.PatchLogger = object
class _CB:
    def __init__(self, *a, **k):
        pass

    def refresh_db_weights(self, *a, **k):
        return None

    def build(self, *a, **k):
        return ""

    def build_context(self, *a, **k):
        return ""

vector_stub.ContextBuilder = _CB
vector_stub.VectorServiceError = Exception
sys.modules.setdefault("menace.vector_service", vector_stub)

import menace.self_coding_engine as self_coding_engine  # noqa: E402
from menace.self_coding_engine import SelfCodingEngine  # noqa: E402


class DummyPromptEngine:
    def __init__(self, *args, tone: str = "neutral", **kwargs):
        self.tone = tone
        self.last_metadata = {}

    def build_prompt(self, description: str, **kwargs) -> Prompt:
        return Prompt(description)


def make_engine(mock_llm, fallback_model: str = "gpt-3.5-turbo"):
    engine = object.__new__(SelfCodingEngine)
    engine.llm_client = mock_llm
    engine.suggest_snippets = MagicMock(return_value=[])
    engine._extract_statements = MagicMock(return_value=[])
    engine._fetch_retry_trace = MagicMock(return_value="")
    engine.prompt_engine = DummyPromptEngine()
    engine.gpt_memory = MagicMock()
    engine.memory_mgr = MagicMock()
    engine.knowledge_service = None
    engine.prompt_tone = "neutral"
    engine.logger = MagicMock()
    engine.simplify_prompt = lambda p: p
    engine._last_prompt_metadata = {}
    engine._last_prompt = None
    engine._last_retry_trace = None
    self_coding_engine._settings = types.SimpleNamespace(codex_retry_delays=[2, 5, 10])
    self_coding_engine.codex_fallback_handler._settings = types.SimpleNamespace(
        codex_fallback_model=fallback_model,
        codex_fallback_strategy="reroute",
    )

    class _CB:
        def __init__(self):
            self.refreshed = False

        def refresh_db_weights(self, *a, **k):
            self.refreshed = True

        def build(self, *a, **k):
            return ""

        def build_context(self, *a, **k):
            return ""

    engine.context_builder = _CB()
    return engine


def patch_history(monkeypatch):
    monkeypatch.setattr(self_coding_engine, "get_feedback", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "get_error_fixes", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "recent_feedback", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_improvement_path", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_error_fix", lambda *a, **k: None)


@pytest.mark.skip(reason="outdated after context builder refactor")
@pytest.mark.skip(reason="outdated after context builder refactor")
def test_empty_output_triggers_fallback(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        Exception("e1"),
        Exception("e2"),
        Exception("e3"),
        LLMResult(text=""),
    ]
    fallback_model = "fallback-a"
    engine = make_engine(mock_llm, fallback_model)

    alt_result = LLMResult(text="print('hi')")
    model_used: dict[str, str] = {}

    def fake_reroute(p: Prompt, *, context_builder=None) -> LLMResult:
        model_used["model"] = (
            self_coding_engine.codex_fallback_handler._settings.codex_fallback_model
        )
        return alt_result

    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler,
        "reroute_to_fallback_model",
        fake_reroute,
    )

    monkeypatch.setattr(time, "sleep", lambda s: None)
    patch_history(monkeypatch)

    manager = types.SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        "do something",
        context_builder=engine.context_builder,
    )

    assert model_used["model"] == fallback_model
    assert result == "print('hi')\n"


@pytest.mark.skip(reason="outdated after context builder refactor")
def test_malformed_output_triggers_fallback(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        Exception("e1"),
        Exception("e2"),
        Exception("e3"),
        LLMResult(text="def bad:"),
    ]
    fallback_model = "fallback-b"
    engine = make_engine(mock_llm, fallback_model)

    alt_result = LLMResult(text="print('fixed')")
    model_used: dict[str, str] = {}

    def fake_reroute(p: Prompt, *, context_builder=None) -> LLMResult:
        model_used["model"] = (
            self_coding_engine.codex_fallback_handler._settings.codex_fallback_model
        )
        return alt_result

    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler,
        "reroute_to_fallback_model",
        fake_reroute,
    )

    monkeypatch.setattr(time, "sleep", lambda s: None)
    patch_history(monkeypatch)

    manager = types.SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        "do something",
        context_builder=engine.context_builder,
    )

    assert model_used["model"] == fallback_model
    assert result == "print('fixed')\n"


def test_fallback_uses_existing_context_builder(monkeypatch):
    mock_llm = MagicMock(return_value=LLMResult(text=""))
    engine = make_engine(mock_llm)
    builder = engine.context_builder

    def handle(prompt, reason, *, context_builder, queue_path=None, **_):
        assert builder.refreshed is True
        assert context_builder is builder
        return LLMResult(text="")

    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler, "handle", handle
    )
    patch_history(monkeypatch)
    manager_generate_helper(
        types.SimpleNamespace(engine=engine),
        "do something",
        context_builder=engine.context_builder,
    )

