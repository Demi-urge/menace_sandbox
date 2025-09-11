import importlib
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest

# Provide a lightweight SelfCodingEngine stub before importing the module
stub_engine_mod = types.ModuleType("menace.self_coding_engine")


class SelfCodingEngine:
    def __init__(self, *a, **k):
        pass


stub_engine_mod.SelfCodingEngine = SelfCodingEngine
sys.modules.setdefault("menace.self_coding_engine", stub_engine_mod)
sys.modules.setdefault("self_coding_engine", stub_engine_mod)

sys.modules.setdefault("menace", types.SimpleNamespace()).RAISE_ERRORS = False

scm_stub = types.ModuleType("menace.self_coding_manager")
class SelfCodingManager:
    def __init__(self, *a, **k):
        pass

    def run_patch(self, path, prompt):  # pragma: no cover - simple stub
        path.write_text("")
scm_stub.SelfCodingManager = SelfCodingManager
sys.modules.setdefault("menace.self_coding_manager", scm_stub)
sys.modules.setdefault("self_coding_manager", scm_stub)

vector_service_mod = types.ModuleType("vector_service")
context_builder_mod = types.ModuleType("vector_service.context_builder")


class ContextBuilder:  # pragma: no cover - simple stub
    pass


class FallbackResult:  # pragma: no cover - simple stub
    pass


class ErrorResult:  # pragma: no cover - simple stub
    pass


context_builder_mod.ContextBuilder = ContextBuilder
context_builder_mod.FallbackResult = FallbackResult
context_builder_mod.ErrorResult = ErrorResult
vector_service_mod.context_builder = context_builder_mod
sys.modules.setdefault("vector_service", vector_service_mod)
sys.modules.setdefault("vector_service.context_builder", context_builder_mod)

sys.modules.setdefault("code_database", types.ModuleType("code_database"))
sys.modules["code_database"].CodeDB = object
sys.modules.setdefault("menace.code_database", sys.modules["code_database"])
sys.modules.setdefault("automated_reviewer", types.ModuleType("automated_reviewer"))
sys.modules["automated_reviewer"].AutomatedReviewer = object
sys.modules.setdefault("menace.automated_reviewer", sys.modules["automated_reviewer"])
sys.modules.setdefault("error_vectorizer", types.SimpleNamespace(ErrorVectorizer=object))
sys.modules.setdefault("failure_fingerprint", types.SimpleNamespace(FailureFingerprint=object))
sys.modules.setdefault(
    "failure_fingerprint_store", types.SimpleNamespace(FailureFingerprintStore=object)
)
vector_service_mod.SharedVectorService = object
sys.modules.setdefault(
    "vector_utils", types.SimpleNamespace(cosine_similarity=lambda *a, **k: 0.0)
)

bot_dev = importlib.import_module("menace.bot_development_bot")
BotDevelopmentBot = bot_dev.BotDevelopmentBot
RetryStrategy = bot_dev.RetryStrategy
BotDevConfig = importlib.import_module("menace.bot_dev_config").BotDevConfig


def make_bot(*, raise_errors: bool = False, attempts: int = 1, with_manager: bool = True):
    bot = BotDevelopmentBot.__new__(BotDevelopmentBot)
    bot.config = BotDevConfig(raise_errors=raise_errors)
    bot.config.max_prompt_log_chars = 1000
    bot.logger = logging.getLogger("test")
    bot._escalate = lambda *a, **k: None
    bot.errors = []
    bot.engine_retry = RetryStrategy(attempts=attempts, delay=0)
    bot.manager = MagicMock() if with_manager else None
    return bot


def test_call_codex_api_success(monkeypatch):
    bot = make_bot()
    monkeypatch.setattr(
        bot_dev,
        "manager_generate_helper",
        lambda mgr, desc: "code",
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do"},
    ]
    result = bot._call_codex_api(messages)
    assert result.success is True
    assert result.code == "code"
    assert result.error is None


def test_call_codex_api_missing_user_prompt(monkeypatch):
    bot = make_bot(raise_errors=True)
    def boom(mgr, desc):  # pragma: no cover - should not run
        raise AssertionError("should not be called")
    monkeypatch.setattr(bot_dev, "manager_generate_helper", boom)
    messages = [{"role": "system", "content": "sys"}]
    with pytest.raises(ValueError):
        bot._call_codex_api(messages)
    assert bot.errors == ["no user prompt provided"]


def test_call_codex_api_engine_failure(monkeypatch):
    bot = make_bot(attempts=3)
    gen = MagicMock(side_effect=Exception("boom"))
    monkeypatch.setattr(bot_dev, "manager_generate_helper", gen)
    messages = [{"role": "user", "content": "do"}]
    result = bot._call_codex_api(messages)
    assert result.success is False
    assert "engine request failed" in result.error
    assert gen.call_count == 3
    assert all(call.args[1] == "user: do" for call in gen.call_args_list)


def test_call_codex_api_no_manager():
    bot = make_bot(with_manager=False)
    messages = [{"role": "user", "content": "do"}]
    result = bot._call_codex_api(messages)
    assert result.success is False
    assert result.error == "SelfCodingManager is required"
