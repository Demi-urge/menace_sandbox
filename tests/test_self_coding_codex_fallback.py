from unittest.mock import MagicMock
import sys
import types
import time

# Stub modules with heavy dependencies before importing llm_interface
rate_limit_stub = sys.modules.setdefault("rate_limit", types.ModuleType("rate_limit"))
rate_limit_stub.estimate_tokens = lambda text: len(str(text))
llm_pricing_stub = sys.modules.setdefault("llm_pricing", types.ModuleType("llm_pricing"))
llm_config_stub = sys.modules.setdefault("llm_config", types.ModuleType("llm_config"))

from llm_interface import LLMResult  # noqa: E402
from prompt_types import Prompt  # noqa: E402

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


def make_engine(mock_llm):
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
    engine._last_prompt_metadata = {}
    engine._last_prompt = None
    engine._last_retry_trace = None
    return engine


def patch_history(monkeypatch):
    monkeypatch.setattr(self_coding_engine, "get_feedback", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "get_error_fixes", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "recent_feedback", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_improvement_path", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_error_fix", lambda *a, **k: None)


def test_empty_output_triggers_fallback(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        Exception("e1"),
        Exception("e2"),
        Exception("e3"),
        LLMResult(text=""),
    ]
    engine = make_engine(mock_llm)

    alt_result = LLMResult(text="print('hi')")
    handle_mock = MagicMock(return_value=alt_result)
    monkeypatch.setattr(self_coding_engine.codex_fallback_handler, "handle_failure", handle_mock)

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    seen_delays: list[list[int]] = []

    def fake_retry(func, *, delays, attempts=None, logger=None, **_kw):
        seen_delays.append(list(delays))
        attempts = attempts or len(delays)
        for i in range(attempts):
            try:
                return func()
            except Exception:
                if i == attempts - 1:
                    raise self_coding_engine.RetryError("boom")
                time.sleep(delays[i])
        raise self_coding_engine.RetryError("boom")

    monkeypatch.setattr(self_coding_engine, "retry_with_backoff", fake_retry)
    patch_history(monkeypatch)

    result = engine.generate_helper("do something")

    assert seen_delays == [[2, 5, 10]]
    assert sleeps == [2, 5]
    assert mock_llm.generate.call_count == 4
    simple_prompt = mock_llm.generate.call_args_list[-1].args[0]
    assert simple_prompt.system == ""
    assert simple_prompt.examples == []
    handle_mock.assert_called_once()
    assert result == "print('hi')\n"


def test_malformed_output_triggers_fallback(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        Exception("e1"),
        Exception("e2"),
        Exception("e3"),
        LLMResult(text="def bad:"),
    ]
    engine = make_engine(mock_llm)

    alt_result = LLMResult(text="print('fixed')")
    handle_mock = MagicMock(return_value=alt_result)
    monkeypatch.setattr(self_coding_engine.codex_fallback_handler, "handle_failure", handle_mock)

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    seen_delays: list[list[int]] = []

    def fake_retry(func, *, delays, attempts=None, logger=None, **_kw):
        seen_delays.append(list(delays))
        attempts = attempts or len(delays)
        for i in range(attempts):
            try:
                return func()
            except Exception:
                if i == attempts - 1:
                    raise self_coding_engine.RetryError("boom")
                time.sleep(delays[i])
        raise self_coding_engine.RetryError("boom")

    monkeypatch.setattr(self_coding_engine, "retry_with_backoff", fake_retry)
    patch_history(monkeypatch)

    result = engine.generate_helper("do something")

    assert seen_delays == [[2, 5, 10]]
    assert sleeps == [2, 5]
    handle_mock.assert_called_once()
    assert result == "print('fixed')\n"
