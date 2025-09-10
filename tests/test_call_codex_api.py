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

bot_dev = importlib.import_module("menace.bot_development_bot")
BotDevelopmentBot = bot_dev.BotDevelopmentBot
RetryStrategy = bot_dev.RetryStrategy
BotDevConfig = importlib.import_module("menace.bot_dev_config").BotDevConfig


def make_bot(*, raise_errors: bool = False, attempts: int = 1):
    bot = BotDevelopmentBot.__new__(BotDevelopmentBot)
    bot.config = BotDevConfig(raise_errors=raise_errors)
    bot.config.max_prompt_log_chars = 1000
    bot.logger = logging.getLogger("test")
    bot._escalate = lambda *a, **k: None
    bot.errors = []
    bot.engine_retry = RetryStrategy(attempts=attempts, delay=0)
    bot.engine = MagicMock()
    return bot


def test_call_codex_api_success():
    bot = make_bot()
    bot.engine.generate_helper.return_value = "code"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do"},
    ]
    result = bot._call_codex_api(messages)
    assert result.success is True
    assert result.code == "code"
    assert result.error is None
    bot.engine.generate_helper.assert_called_once_with("system: sys\nuser: do")


def test_call_codex_api_missing_user_prompt():
    bot = make_bot(raise_errors=True)
    messages = [{"role": "system", "content": "sys"}]
    with pytest.raises(ValueError):
        bot._call_codex_api(messages)
    bot.engine.generate_helper.assert_not_called()


def test_call_codex_api_engine_failure():
    bot = make_bot(attempts=3)
    bot.engine.generate_helper.side_effect = Exception("boom")
    messages = [{"role": "user", "content": "do"}]
    result = bot._call_codex_api(messages)
    assert result.success is False
    assert "engine request failed" in result.error
    assert bot.engine.generate_helper.call_count == 3
    assert all(
        call.args[0] == "user: do" for call in bot.engine.generate_helper.call_args_list
    )
