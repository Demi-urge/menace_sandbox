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

db_router_stub = types.ModuleType("db_router")
db_router_stub.DBRouter = object
db_router_stub.init_db_router = lambda *a, **k: None
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.SHARED_TABLES = {}
db_router_stub.queue_insert = lambda *a, **k: None
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
vector_service.SharedVectorService = object
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

chatgpt_enh = types.ModuleType("chatgpt_enhancement_bot")
chatgpt_enh.EnhancementDB = type("EnhancementDB", (), {})
chatgpt_enh.EnhancementHistory = type("EnhancementHistory", (), {})
chatgpt_enh.Enhancement = type("Enhancement", (), {})
sys.modules.setdefault("chatgpt_enhancement_bot", chatgpt_enh)
sys.modules.setdefault("menace_sandbox.chatgpt_enhancement_bot", chatgpt_enh)
sys.modules.setdefault("error_vectorizer", types.SimpleNamespace(ErrorVectorizer=object))
sys.modules.setdefault("failure_fingerprint", types.SimpleNamespace(FailureFingerprint=object))
sys.modules.setdefault(
    "failure_fingerprint_store", types.SimpleNamespace(FailureFingerprintStore=object)
)
sys.modules.setdefault(
    "vector_utils", types.SimpleNamespace(cosine_similarity=lambda *a, **k: 0.0)
)

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

scm_stub = types.ModuleType("menace_sandbox.self_coding_manager")
class SelfCodingManager:
    def __init__(self, *a, **k):
        pass

    def run_patch(self, path, prompt):
        path.write_text("")
scm_stub.SelfCodingManager = SelfCodingManager
sys.modules.setdefault("menace_sandbox.self_coding_manager", scm_stub)
sys.modules.setdefault("self_coding_manager", scm_stub)

from menace_sandbox.bot_development_bot import (
    BotDevelopmentBot,
    RetryStrategy,
    EngineResult,
)
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

    def fake_generate(_mgr, desc: str) -> str:
        captured["desc"] = desc
        return "code"

    def fake_wrapper(*a, **k):
        nonlocal wrapper_called
        wrapper_called = True
        return {}

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    monkeypatch.setattr(memory_client, "ask_with_memory", fake_wrapper)
    monkeypatch.setattr(
        RetryStrategy,
        "run",
        lambda self, func, logger=None: func(),
    )

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
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

    assert captured["desc"] == "system: sys\nuser: hi\nassistant: there\nuser: again"
    assert result == EngineResult(True, "code", None)
    assert wrapper_called is False


def test_call_codex_api_aggregates_multi_messages(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate(_mgr, desc: str) -> str:
        captured["desc"] = desc
        return ""  # pragma: no cover - return value unused

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    monkeypatch.setattr(
        RetryStrategy,
        "run",
        lambda self, func, logger=None: func(),
    )

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    BotDevelopmentBot._call_codex_api(
        dummy,
        [
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "u2"},
        ],
    )

    assert (
        captured["desc"]
        == "system: s1\nuser: u1\nassistant: a1\ntool: t1\nsystem: s2\nuser: u2"
    )


def test_call_codex_api_includes_tool_messages(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate(_mgr, desc: str) -> str:
        captured["desc"] = desc
        return ""  # pragma: no cover - return value unused

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    monkeypatch.setattr(
        RetryStrategy, "run", lambda self, func, logger=None: func()
    )

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    BotDevelopmentBot._call_codex_api(
        dummy,
        [
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "t"},
            {"role": "assistant", "content": "a"},
        ],
    )

    assert captured["desc"] == "user: u\ntool: t\nassistant: a"


def test_call_codex_api_no_user_message_escalates(monkeypatch, caplog):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_mgr, _desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "assistant", "content": "there"}],
        )

    assert result == EngineResult(False, None, "no user prompt provided")
    assert escalated["level"] == "warning"
    assert "no user prompt provided" in escalated["msg"]
    assert "no user prompt provided" in caplog.text
    assert dummy.errors == ["no user prompt provided"]
    assert run_called is False


def test_call_codex_api_no_user_message_raises_value_error(monkeypatch):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_mgr, _desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=True),
    )

    with pytest.raises(ValueError):
        BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "assistant", "content": "there"}],
        )

    assert escalated["level"] == "warning"
    assert dummy.errors == ["no user prompt provided"]
    assert run_called is False


def test_call_codex_api_empty_messages_escalates(monkeypatch, caplog):
    escalated: dict[str, str] = {}

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_mgr, _desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
    run_called = False

    def fake_run(self, func, logger=None):
        nonlocal run_called
        run_called = True
        return func()

    monkeypatch.setattr(RetryStrategy, "run", fake_run)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(dummy, [])

    assert result == EngineResult(False, None, "no user prompt provided")
    assert escalated["level"] == "warning"
    assert "no user prompt provided" in escalated["msg"]
    assert "no user prompt provided" in caplog.text
    assert dummy.errors == ["no user prompt provided"]
    assert run_called is False


def test_call_codex_api_engine_failure_retries_and_escalates(monkeypatch, caplog):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def fake_generate(_mgr, _desc: str) -> str:
        calls["n"] += 1
        raise RuntimeError("boom")

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)
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
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    with caplog.at_level(logging.WARNING):
        result = BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "user", "content": "hi"}],
        )

    assert calls["n"] == 2
    assert result == EngineResult(False, None, "engine request failed: boom")
    assert escalated["msg"] == "engine request failed: boom"
    assert escalated["level"] == "error"
    assert dummy.errors == ["engine request failed: boom"]
    assert "retry 1/2 after error: boom" in caplog.text


def test_call_codex_api_logs_prompt_and_handles_exception(monkeypatch, caplog):
    escalated: dict[str, str] = {}

    def boom(_mgr, _desc: str) -> str:
        raise RuntimeError("kaboom")

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", boom)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": escalated.update({"msg": msg, "level": level}),
        errors=[],
        engine_retry=RetryStrategy(attempts=1, delay=0),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    prompt = "hello " * 50

    with caplog.at_level(logging.INFO):
        result = BotDevelopmentBot._call_codex_api(
            dummy, [{"role": "user", "content": prompt}]
        )

    assert "generate_helper prompt" in caplog.text
    assert ("user: " + prompt)[:200] in caplog.text
    assert result == EngineResult(False, None, "engine request failed: kaboom")
    assert escalated["msg"] == "engine request failed: kaboom"
    assert dummy.errors == ["engine request failed: kaboom"]


def test_call_codex_api_scrubs_secrets_from_prompt(monkeypatch, caplog):
    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", lambda _mgr, _d: "ok")
    monkeypatch.setattr(RetryStrategy, "run", lambda self, func, logger=None: func())

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    prompt = "api_key=SECRET token=VALUE"
    with caplog.at_level(logging.INFO):
        BotDevelopmentBot._call_codex_api(
            dummy, [{"role": "user", "content": prompt}]
        )

    assert "[REDACTED]" in caplog.text
    assert "SECRET" not in caplog.text
    assert "VALUE" not in caplog.text


def test_call_codex_api_respects_log_limit(monkeypatch, caplog):
    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", lambda _mgr, _d: "ok")
    monkeypatch.setattr(RetryStrategy, "run", lambda self, func, logger=None: func())

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=10, raise_errors=False),
    )

    prompt = "x" * 50
    with caplog.at_level(logging.INFO):
        BotDevelopmentBot._call_codex_api(
            dummy, [{"role": "user", "content": prompt}]
        )

    assert "user: " + "x" * 4 in caplog.text
    assert "user: " + "x" * 5 not in caplog.text


def test_call_codex_api_engine_failure_raises(monkeypatch):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def fake_generate(_mgr, _desc: str) -> str:
        calls["n"] += 1
        raise RuntimeError("boom")

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)

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
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=True),
    )

    with pytest.raises(RuntimeError):
        BotDevelopmentBot._call_codex_api(
            dummy,
            [{"role": "user", "content": "hi"}],
        )

    assert calls["n"] == 2
    assert escalated["msg"] == "engine request failed: boom"
    assert escalated["level"] == "error"
    assert dummy.errors == ["engine request failed: boom"]


def test_call_codex_api_retries_then_succeeds(monkeypatch):
    calls: dict[str, int] = {"n": 0}
    escalated: dict[str, str] = {}

    def flaky_generate(_mgr, _desc: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky")
        return "ok"

    def fake_escalate(msg: str, level: str = "error") -> None:
        escalated["msg"] = msg
        escalated["level"] = level

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(bdb, "manager_generate_helper", flaky_generate)

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=fake_escalate,
        errors=[],
        engine_retry=RetryStrategy(attempts=2, delay=0),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [{"role": "user", "content": "hi"}],
    )

    assert result == EngineResult(True, "ok", None)
    assert calls["n"] == 2
    assert escalated == {}
    assert dummy.errors == []


def test_call_codex_api_missing_user_message(monkeypatch):
    """_call_codex_api should escalate when no user prompt is supplied."""

    escalated: dict[str, str] = {}
    engine = sce_stub.SelfCodingEngine()

    def fake_generate(_mgr, _desc: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("generate_helper should not be called")

    monkeypatch.setattr(bdb, "manager_generate_helper", fake_generate)

    class FakeRetry:
        def __init__(self) -> None:
            self.called = False

        def run(self, func, logger=None):  # pragma: no cover - should not run
            self.called = True
            return func()

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": escalated.update({"msg": msg, "level": level}),
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    fake_retry = FakeRetry()
    monkeypatch.setattr(dummy, "engine_retry", fake_retry)

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [{"role": "assistant", "content": "there"}],
    )

    assert result == EngineResult(False, None, "no user prompt provided")
    assert escalated["level"] == "warning"
    assert escalated["msg"] == "no user prompt provided"
    assert fake_retry.called is False


def test_call_codex_api_generate_helper_exception(monkeypatch):
    """Errors from generate_helper should be captured and returned."""

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(
        bdb,
        "manager_generate_helper",
        lambda _mgr, _desc: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    class FakeRetry:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, func, logger=None):
            self.calls += 1
            return func()

    fake_retry = FakeRetry()

    escalated: dict[str, str] = {}

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": escalated.update({"msg": msg, "level": level}),
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    monkeypatch.setattr(dummy, "engine_retry", fake_retry)

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [{"role": "user", "content": "hi"}],
    )

    assert result == EngineResult(False, None, "engine request failed: boom")
    assert fake_retry.calls == 1
    assert escalated["msg"] == "engine request failed: boom"
    assert dummy.errors == ["engine request failed: boom"]


def test_call_codex_api_retry_succeeds(monkeypatch):
    """Retrying generate_helper should succeed after an initial failure."""

    engine = sce_stub.SelfCodingEngine()
    calls: dict[str, int] = {"gen": 0, "run": 0}

    def flaky_generate(_mgr, _desc: str) -> str:
        calls["gen"] += 1
        if calls["gen"] == 1:
            raise RuntimeError("flaky")
        return "ok"

    monkeypatch.setattr(bdb, "manager_generate_helper", flaky_generate)

    class FakeRetry:
        def run(self, func, logger=None):
            for _ in range(2):
                calls["run"] += 1
                try:
                    return func()
                except Exception:
                    if _ == 1:
                        raise

    fake_retry = FakeRetry()

    dummy = types.SimpleNamespace(
        coding_engine=engine,
        engine=engine,
        manager=types.SimpleNamespace(engine=engine),
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
        engine_retry=RetryStrategy(),
        config=types.SimpleNamespace(max_prompt_log_chars=200, raise_errors=False),
    )

    monkeypatch.setattr(dummy, "engine_retry", fake_retry)

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        [{"role": "user", "content": "hi"}],
    )

    assert result == EngineResult(True, "ok", None)
    assert calls == {"gen": 2, "run": 2}
    assert dummy.errors == []
