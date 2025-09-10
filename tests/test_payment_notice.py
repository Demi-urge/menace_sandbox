"""Tests for legacy payment notice injection logic.

SelfCodingEngine now handles all code generation locally, but the helper
is retained for backwards compatibility.
"""

# flake8: noqa

import sys
import types
from pathlib import Path
import logging

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from billing.prompt_notice import PAYMENT_ROUTER_NOTICE, prepend_payment_notice

# Stub heavy dependencies so chatgpt_idea_bot can be imported without side effects
package = types.ModuleType("menace_sandbox")
package.RAISE_ERRORS = False
sys.modules.setdefault("menace_sandbox", package)

dm = types.ModuleType("menace_sandbox.database_manager")
dm.DB_PATH = "db"
dm.search_models = lambda *a, **k: []
sys.modules["menace_sandbox.database_manager"] = dm

dmb = types.ModuleType("menace_sandbox.database_management_bot")
class _DBot: ...
dmb.DatabaseManagementBot = _DBot
sys.modules["menace_sandbox.database_management_bot"] = dmb

log_tags = types.SimpleNamespace(
    FEEDBACK="fb", IMPROVEMENT_PATH="ip", ERROR_FIX="ef", INSIGHT="ins"
)
sys.modules["menace_sandbox.log_tags"] = log_tags

shared_mem = types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
sys.modules["menace_sandbox.shared_gpt_memory"] = shared_mem

memory_logging = types.SimpleNamespace(log_with_tags=lambda *a, **k: None)
sys.modules["menace_sandbox.memory_logging"] = memory_logging

memory_client = types.SimpleNamespace(ask_with_memory=lambda *a, **k: {})
sys.modules["menace_sandbox.memory_aware_gpt_client"] = memory_client

db_router_stub = types.SimpleNamespace(
    DBRouter=object(), init_db_router=lambda *a, **k: None, GLOBAL_ROUTER=None
)
sys.modules["db_router"] = db_router_stub
sys.modules["menace_sandbox.db_router"] = db_router_stub
sys.modules["menace_sandbox"].db_router = db_router_stub  # type: ignore[attr-defined]

class _LKM:
    def __init__(self, *a, **k):
        pass
local_mod = types.SimpleNamespace(LocalKnowledgeModule=_LKM)
sys.modules["menace_sandbox.local_knowledge_module"] = local_mod

retriever_stub = types.SimpleNamespace(
    get_feedback=lambda *a, **k: [],
    get_improvement_paths=lambda *a, **k: [],
    get_error_fixes=lambda *a, **k: [],
)
sys.modules["menace_sandbox.knowledge_retriever"] = retriever_stub

sys.modules["menace_sandbox.run_autonomous"] = types.SimpleNamespace(
    LOCAL_KNOWLEDGE_MODULE=None
)
sys.modules["menace_sandbox.sandbox_runner"] = types.SimpleNamespace(
    LOCAL_KNOWLEDGE_MODULE=None
)

class _CB:
    def __init__(self, *a, **k):
        self.called = False

    def build(self, *a, **k):
        self.called = True
        return ""

vector_service = types.ModuleType("vector_service")
vector_service.Retriever = None
vector_service.FallbackResult = list
vector_service.ErrorResult = Exception
vector_service.ContextBuilder = _CB
vector_service.CognitionLayer = object
vector_service.__path__ = []  # type: ignore[attr-defined]
vector_service.__spec__ = types.SimpleNamespace(submodule_search_locations=[])  # type: ignore[attr-defined]
sys.modules.setdefault("vector_service", vector_service)
vec_cb = types.ModuleType("vector_service.context_builder")
vec_cb.ContextBuilder = _CB
vec_cb.FallbackResult = list
vec_cb.ErrorResult = Exception
sys.modules.setdefault("vector_service.context_builder", vec_cb)
vec_ret = types.ModuleType("vector_service.retriever")
vec_ret.Retriever = None
vec_ret.FallbackResult = list
sys.modules.setdefault("vector_service.retriever", vec_ret)
vec_roi = types.ModuleType("vector_service.roi_tags")
vec_roi.RoiTag = type("RoiTag", (), {})
sys.modules.setdefault("vector_service.roi_tags", vec_roi)

governed = types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x)
sys.modules.setdefault("governed_retrieval", governed)

# Stubs for enhancement_bot dependencies
code_db = types.SimpleNamespace(CodeDB=type("CodeDB", (), {}))
sys.modules.setdefault("code_database", code_db)
sys.modules.setdefault("menace_sandbox.code_database", code_db)

chatgpt_enh = types.SimpleNamespace(
    EnhancementDB=type("EnhancementDB", (), {}),
    EnhancementHistory=type("EnhancementHistory", (), {}),
    Enhancement=type("Enhancement", (), {}),
)
sys.modules.setdefault("chatgpt_enhancement_bot", chatgpt_enh)
sys.modules.setdefault("menace_sandbox.chatgpt_enhancement_bot", chatgpt_enh)

diff_summarizer = types.SimpleNamespace(summarize_diff=lambda *a, **k: "")
prefix_injector = types.SimpleNamespace(
    inject_prefix=lambda msgs, prefix, conf, role="system": msgs
)
micro_models_pkg = types.ModuleType("menace_sandbox.micro_models")
sys.modules.setdefault("menace_sandbox.micro_models", micro_models_pkg)
sys.modules.setdefault(
    "menace_sandbox.micro_models.diff_summarizer", diff_summarizer
)
sys.modules.setdefault(
    "menace_sandbox.micro_models.prefix_injector", prefix_injector
)

sce_stub = types.ModuleType("menace_sandbox.self_coding_engine")


class _DummyEngine:
    def __init__(self, *a, **k):
        pass

    def generate_helper(self, desc: str) -> str:
        return ""


sce_stub.SelfCodingEngine = _DummyEngine
sys.modules.setdefault("menace_sandbox.self_coding_engine", sce_stub)
sys.modules.setdefault("self_coding_engine", sce_stub)

from menace_sandbox.bot_development_bot import BotDevelopmentBot, RetryStrategy
import menace_sandbox.bot_development_bot as bdb


def test_payment_router_notice_mentions_central_routing_and_logging():
    phrase = (
        "Every Stripe charge must use central routing and log via billing_logger/stripe_ledger."
    )
    assert phrase in PAYMENT_ROUTER_NOTICE


def test_prepend_payment_notice_helper():
    msgs = [{"role": "user", "content": "hello"}]
    new_msgs = prepend_payment_notice(msgs)
    assert new_msgs[0]["role"] == "system"
    assert new_msgs[1]["content"] == "hello"


def test_call_codex_api_forwards_prompt_to_engine(monkeypatch):
    captured: dict[str, str] = {}
    wrapper_called = False

    def fake_generate(desc: str) -> str:
        captured["desc"] = desc
        return "code"

    def fake_wrapper(*a, **k):
        nonlocal wrapper_called
        wrapper_called = True
        return {}

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(memory_client, "ask_with_memory", fake_wrapper)
    monkeypatch.setattr(
        RetryStrategy,
        "run",
        lambda self, func, logger=None: func(),
    )

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
    )

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
            {"role": "user", "content": "again"},
        ],
    )

    assert captured["desc"] == "system: sys\nuser: hi\nuser: again"
    assert result == "code"
    assert wrapper_called is False


def test_call_codex_api_aggregates_multi_messages(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate(desc: str) -> str:
        captured["desc"] = desc
        return ""  # pragma: no cover - return value unused

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(
        RetryStrategy,
        "run",
        lambda self, func, logger=None: func(),
    )

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
    )

    BotDevelopmentBot._call_codex_api(
        dummy,
        [
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "u2"},
        ],
    )

    assert captured["desc"] == "system: s1\nuser: u1\nsystem: s2\nuser: u2"


def test_call_codex_api_no_user_message_escalates(monkeypatch, caplog):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", False)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "assistant", "content": "there"}],
        )

    assert result is None
    assert escalated["level"] == "warning"
    assert "no user message found" in escalated["msg"]
    assert "no user message found" in caplog.text
    assert run_called is False


def test_call_codex_api_no_user_message_raises_value_error(monkeypatch):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", True)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
    )

    with pytest.raises(ValueError):
        BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "assistant", "content": "there"}],
        )

    assert escalated["level"] == "warning"
    assert run_called is False


def test_call_codex_api_empty_messages_escalates(monkeypatch, caplog):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", False)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(dummy, [])

    assert result is None
    assert escalated["level"] == "warning"
    assert "no user message found" in escalated["msg"]
    assert "no user message found" in caplog.text
    assert run_called is False


def test_call_codex_api_engine_failure_retries_and_escalates(monkeypatch, caplog):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def fake_generate(_desc: str) -> str:
        calls["n"] += 1
        raise RuntimeError("boom")

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", False)
    
    def fake_run(self, func, logger=None):
        for i in range(self.attempts):
            try:
                return func()
            except Exception as exc:
                if logger:
                    logger.warning(
                        "retry %s/%s after error: %s", i + 1, self.attempts, exc
                    )
                if i == self.attempts - 1:
                    raise

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "user", "content": "hi"}],
        )

    assert calls["n"] == 2
    assert result == {"error": "engine request failed after retries: boom"}
    assert escalated["msg"] == "engine request failed after retries: boom"
    assert escalated["level"] == "error"
    assert dummy.errors == ["engine request failed after retries: boom"]
    assert "retry 1/2 after error: boom" in caplog.text


def test_call_codex_api_engine_failure_raises(monkeypatch):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def fake_generate(_desc: str) -> str:
        calls["n"] += 1
        raise RuntimeError("boom")

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", fake_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", True)
    
    def fake_run(self, func, logger=None):
        for i in range(self.attempts):
            try:
                return func()
            except Exception as exc:
                if logger:
                    logger.warning(
                        "retry %s/%s after error: %s", i + 1, self.attempts, exc
                    )
                if i == self.attempts - 1:
                    raise

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
    )

    with pytest.raises(RuntimeError):
        BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "user", "content": "hi"}],
        )

    assert calls["n"] == 2
    assert escalated["msg"] == "engine request failed after retries: boom"
    assert escalated["level"] == "error"
    assert dummy.errors == ["engine request failed after retries: boom"]


def test_call_codex_api_retries_then_succeeds(monkeypatch):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def flaky_generate(_desc: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky")
        return "ok"

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", flaky_generate)
    monkeypatch.setattr(bdb, "RAISE_ERRORS", False)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
    )

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [{"role": "user", "content": "hi"}],
    )

    assert result == "ok"
    assert calls["n"] == 2
    assert escalated == {}
    assert dummy.errors == []
