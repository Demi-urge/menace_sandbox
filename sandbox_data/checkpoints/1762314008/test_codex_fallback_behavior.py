import pytest
from unittest.mock import MagicMock
import sys
import types

from llm_interface import LLMResult
from prompt_types import Prompt
from menace.coding_bot_interface import manager_generate_helper

pytestmark = pytest.mark.skip(reason="context builder refactor")

# Stub heavy dependencies before importing self_coding_engine
code_db_stub = sys.modules.setdefault('code_database', types.ModuleType('code_database'))
code_db_stub.CodeDB = object
code_db_stub.CodeRecord = object
code_db_stub.PatchHistoryDB = object
code_db_stub.PatchRecord = object
sys.modules.setdefault('menace.code_database', code_db_stub)

ue_stub = sys.modules.setdefault('unified_event_bus', types.ModuleType('unified_event_bus'))
ue_stub.UnifiedEventBus = object
sys.modules.setdefault('menace.unified_event_bus', ue_stub)

sgm_stub = sys.modules.setdefault('shared_gpt_memory', types.ModuleType('shared_gpt_memory'))
sgm_stub.GPT_MEMORY_MANAGER = object
sys.modules.setdefault('menace.shared_gpt_memory', sgm_stub)

gpt_mem_stub = sys.modules.setdefault('gpt_memory', types.ModuleType('gpt_memory'))
gpt_mem_stub.GPTMemoryManager = object
gpt_mem_stub.INSIGHT = 'INSIGHT'
gpt_mem_stub._summarise_text = lambda text, *a, **k: text
sys.modules.setdefault('menace.gpt_memory', gpt_mem_stub)

vector_stub = sys.modules.setdefault('vector_service', types.ModuleType('vector_service'))
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
sys.modules.setdefault('menace.vector_service', vector_stub)

import menace.self_coding_engine as self_coding_engine  # noqa: E402
from menace.self_coding_engine import SelfCodingEngine  # noqa: E402


class DummyPromptEngine:
    def __init__(self, *args, tone: str = 'neutral', **kwargs):
        self.tone = tone
        self.last_metadata = {}

    def build_prompt(self, description: str, **kwargs) -> Prompt:
        return Prompt(description, system='sys', examples=['ex1', 'ex2'])


def make_engine(mock_llm, monkeypatch):
    engine = object.__new__(SelfCodingEngine)
    engine.llm_client = mock_llm
    engine.suggest_snippets = MagicMock(return_value=[])
    engine._extract_statements = MagicMock(return_value=[])
    engine._fetch_retry_trace = MagicMock(return_value='')
    engine.prompt_engine = DummyPromptEngine()
    engine.gpt_memory = MagicMock()
    engine.memory_mgr = MagicMock()
    engine.knowledge_service = None
    engine.prompt_tone = 'neutral'
    engine.logger = MagicMock()
    engine._last_prompt_metadata = {}
    engine._last_prompt = None
    engine._last_retry_trace = None
    engine.simplify_prompt = self_coding_engine.simplify_prompt
    monkeypatch.setattr(
        self_coding_engine, '_settings', types.SimpleNamespace(codex_retry_delays=[2, 5, 10])
    )
    monkeypatch.setattr(
        self_coding_engine,
        'create_context_builder',
        lambda: types.SimpleNamespace(
            refresh_db_weights=lambda *a, **k: None, build=lambda *a, **k: ""
        ),
    )
    return engine


def patch_history(monkeypatch):
    monkeypatch.setattr(self_coding_engine, 'get_feedback', lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, 'get_error_fixes', lambda *a, **k: [])
    monkeypatch.setattr(self_coding_engine, 'recent_feedback', lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, 'recent_improvement_path', lambda *a, **k: None)
    monkeypatch.setattr(self_coding_engine, 'recent_error_fix', lambda *a, **k: None)


def _expected_fallback(desc: str) -> str:
    return (
        f"def auto_{desc.replace(' ', '_')}(*args, **kwargs):\n"
        f'    """{desc}"""\n'
        "    return {\n"
        f"        'description': '{desc}',\n"
        "        'args': args,\n"
        "        'kwargs': kwargs,\n"
        "    }\n"
    )


def test_timeout_error_prompts_simplified_and_builtin_fallback(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = TimeoutError('boom')
    engine = make_engine(mock_llm, monkeypatch)

    calls = []

    def fake_call(client, prompt, *, context_builder=None, logger=None, timeout=30.0):
        calls.append(prompt)
        delays = list(self_coding_engine._settings.codex_retry_delays)
        for _ in delays:
            try:
                client.generate(prompt)
            except Exception:
                continue
        raise self_coding_engine.RetryError('timeout')

    monkeypatch.setattr(self_coding_engine, 'call_codex_with_backoff', fake_call)

    handle_mock = MagicMock(return_value=LLMResult(text=''))
    monkeypatch.setattr(self_coding_engine.codex_fallback_handler, 'handle', handle_mock)
    patch_history(monkeypatch)

    manager = types.SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        'do something',
        context_builder=engine.context_builder,
    )

    assert len(calls) == 2
    assert calls[1].system == ''
    assert calls[1].examples == []
    assert mock_llm.generate.call_count == 6


def test_strip_prompt_context_clears_context():
    prompt = Prompt('user request', system='sys', examples=['a', 'b'])
    simplified = self_coding_engine.strip_prompt_context(prompt)
    assert simplified.system == ''
    assert simplified.examples == []
    assert simplified.user == 'user request'


def test_empty_completion_reroutes_and_queues(monkeypatch):
    mock_llm = MagicMock(return_value=LLMResult(text=''))
    engine = make_engine(mock_llm, monkeypatch)

    self_coding_engine.codex_fallback_handler._settings = types.SimpleNamespace(
        codex_fallback_strategy="reroute"
    )

    def simple_call(
        client, prompt, *, context_builder=None, logger=None, timeout=30.0
    ):
        return client.generate(prompt)

    monkeypatch.setattr(self_coding_engine, 'call_codex_with_backoff', simple_call)

    q_mock = MagicMock()
    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler, 'queue_failed', q_mock
    )

    def boom(_prompt, *, context_builder=None):
        raise RuntimeError('fail')

    monkeypatch.setattr(
        self_coding_engine.codex_fallback_handler, 'reroute_to_fallback_model', boom
    )

    patch_history(monkeypatch)

    manager = types.SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        'do something',
        context_builder=engine.context_builder,
    )

    q_mock.assert_called_once()
    assert result == _expected_fallback('do something')


def test_handle_returns_llmresult_used_by_engine(monkeypatch):
    mock_llm = MagicMock(return_value=LLMResult(text=''))
    engine = make_engine(mock_llm, monkeypatch)

    def simple_call(
        client, prompt, *, context_builder=None, logger=None, timeout=30.0
    ):
        return client.generate(prompt)

    monkeypatch.setattr(self_coding_engine, 'call_codex_with_backoff', simple_call)

    alt = LLMResult(text="print('hi')")
    handle_mock = MagicMock(return_value=alt)
    monkeypatch.setattr(self_coding_engine.codex_fallback_handler, 'handle', handle_mock)

    patch_history(monkeypatch)

    manager = types.SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        'do something',
        context_builder=engine.context_builder,
    )

    handle_mock.assert_called_once()
    assert isinstance(handle_mock.return_value, LLMResult)
    assert result == "print('hi')\n"
