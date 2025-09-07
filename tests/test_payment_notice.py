"""Tests for payment notice injection logic."""

# flake8: noqa

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from billing.prompt_notice import PAYMENT_ROUTER_NOTICE, prepend_payment_notice
from prompt_engine import PromptEngine

from billing.openai_wrapper import chat_completion_create

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

vector_service = types.SimpleNamespace(
    Retriever=None,
    FallbackResult=list,
    ErrorResult=Exception,
    ContextBuilder=type("ContextBuilder", (), {}),
)
sys.modules.setdefault("vector_service", vector_service)

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

diff_summarizer = types.SimpleNamespace(summarize_diff=lambda a, b: "")
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

from menace_sandbox.enhancement_bot import EnhancementBot
from menace_sandbox.chatgpt_idea_bot import ChatGPTClient
from menace_sandbox.bot_development_bot import BotDevelopmentBot


def test_payment_router_notice_mentions_central_routing_and_logging():
    phrase = (
        "Every Stripe charge must use central routing and log via billing_logger/stripe_ledger."
    )
    assert phrase in PAYMENT_ROUTER_NOTICE


def test_prepend_payment_notice_helper():
    msgs = [{"role": "user", "content": "hello"}]
    new_msgs = prepend_payment_notice(msgs)
    assert new_msgs[0]["role"] == "system"
    assert new_msgs[0]["content"].startswith(PAYMENT_ROUTER_NOTICE)
    assert new_msgs[1]["content"] == "hello"


def test_chatgpt_client_injects_notice(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            return None

    class DummySession:
        def post(self, url, headers=None, json=None, timeout=None):
            captured["messages"] = json["messages"]
            return DummyResponse()

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = ChatGPTClient(session=DummySession(), gpt_memory=None, context_builder=DummyBuilder())
    client.ask([{"role": "user", "content": "hi"}], use_memory=False, tags=[])
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)


def test_enhancement_bot_injects_notice(monkeypatch):
    captured = {}
    fake_openai = types.SimpleNamespace()

    def fake_create(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return {"choices": [{"message": {"content": ""}}]}

    fake_openai.ChatCompletion = types.SimpleNamespace(create=fake_create)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    bot = EnhancementBot()
    bot._codex_summarize("a", "b")
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)


def test_prompt_engine_build_prompt_contains_notice():
    engine = PromptEngine(retriever=None)
    prompt = engine.build_prompt("task")
    assert prompt.system.startswith(PAYMENT_ROUTER_NOTICE)


def test_bot_development_bot_injects_notice(monkeypatch):
    captured = {}
    fake_openai = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(
            create=lambda *a, **k: captured.update(messages=k.get("messages")) or {}
        )
    )
    monkeypatch.setattr(
        "menace_sandbox.bot_development_bot.openai", fake_openai, raising=False
    )
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    BotDevelopmentBot._call_codex_api(object(), "m", [{"role": "user", "content": "hi"}])
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)


def test_openai_wrapper_injects_notice():
    captured = {}

    def fake_create(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return {}

    fake_openai = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=fake_create)
    )
    chat_completion_create(
        [{"role": "user", "content": "hi"}],
        model="gpt-3.5-turbo",
        openai_client=fake_openai,
    )
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)


def test_gpt4client_injects_notice(monkeypatch):
    captured = {}

    def fake_create(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return iter([{ "choices": [{"delta": {"content": ""}}] }])

    fake_openai = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=fake_create)
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "stripe_billing_router", types.ModuleType("sbr"))
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "neurosales")
    )
    import importlib
    ext = importlib.import_module("neurosales.external_integrations")
    importlib.reload(ext)
    from neurosales.external_integrations import GPT4Client

    client = GPT4Client(api_key="k")
    list(client.stream_chat("arch", [0.1], "obj", "hi"))
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)
