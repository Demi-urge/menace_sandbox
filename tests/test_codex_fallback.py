from unittest.mock import MagicMock
import sys
import types

from llm_interface import LLMResult
from prompt_types import Prompt

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
        return Prompt(description, system="sys", examples=["ex1", "ex2"])


def make_engine(mock_llm, monkeypatch):
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
    engine.simplify_prompt = self_coding_engine.simplify_prompt
    monkeypatch.setattr(
        self_coding_engine, "_settings", types.SimpleNamespace(codex_retry_delays=[2, 5, 10])
    )
    monkeypatch.setattr(
        self_coding_engine,
        "create_context_builder",
        lambda: types.SimpleNamespace(
            refresh_db_weights=lambda *a, **k: None, build=lambda *a, **k: ""
        ),
    )
    return engine


def patch_history(monkeypatch):
    monkeypatch.setattr(self_coding_engine, "get_feedback", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "get_error_fixes", lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, "recent_feedback", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_improvement_path", lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, "recent_error_fix", lambda *a, **k: None)


def test_codex_fallback_retries_and_simplified_prompt(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        Exception("e1"),
        Exception("e2"),
        Exception("e3"),
        Exception("e4"),
        Exception("e5"),
        Exception("e6"),
    ]
    engine = make_engine(mock_llm, monkeypatch)

    alt_result = LLMResult(text="print('hi')")
    handle_mock = MagicMock(return_value=alt_result)
    monkeypatch.setattr(self_coding_engine.codex_fallback_handler, "handle", handle_mock)

    call_delays = []

    def fake_call(client, prompt, *, logger=None, timeout=30.0):
        delays = list(self_coding_engine._settings.codex_retry_delays)
        call_delays.append(delays)
        for _ in delays:
            try:
                client.generate(prompt)
            except Exception:
                continue
        raise self_coding_engine.RetryError("boom")

    monkeypatch.setattr(self_coding_engine, "call_codex_with_backoff", fake_call)
    patch_history(monkeypatch)

    result = engine.generate_helper("do something")

    assert call_delays == [[2, 5, 10], [2, 5, 10]]
    assert mock_llm.generate.call_count == 6
    simple_prompt = mock_llm.generate.call_args_list[-1].args[0]
    assert simple_prompt.system == ""
    assert simple_prompt.examples == []
    handle_mock.assert_called_once()
    assert result == "print('hi')\n"


def test_codex_fallback_queue_on_malformed(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResult(text="def bad:")
    engine = make_engine(mock_llm, monkeypatch)

    queue_mock = MagicMock()
    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler,
        "queue_for_retry",
        queue_mock,
        raising=False,
    )

    def handle(prompt, reason, *, context_builder, **_):
        self_coding_engine.codex_fallback_handler.queue_for_retry(prompt)
        return LLMResult(text="")

    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler, "handle", handle
    )
    patch_history(monkeypatch)

    engine.generate_helper("do something")
    queue_mock.assert_called_once()
