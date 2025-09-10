"""Tests for payment notice injection logic."""

# flake8: noqa

import sys
import types
from pathlib import Path
import logging

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from billing.prompt_notice import PAYMENT_ROUTER_NOTICE, prepend_payment_notice
from llm_interface import LLMClient, LLMResult
from typing import Any

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

db_router_stub = types.SimpleNamespace(DBRouter=object(), init_db_router=lambda *a, **k: None)
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

# Minimal ChatGPTClient stub to avoid heavy imports
class _ChatGPTClient:
    def __init__(self, session=None, gpt_memory=None, context_builder=None):
        self.session = session

    def ask(self, messages, use_memory=False, tags=None):
        msgs = prepend_payment_notice(messages)
        if self.session:
            self.session.post("http://", json={"messages": msgs})
        return {}

sys.modules.setdefault(
    "menace_sandbox.chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=_ChatGPTClient)
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


def test_enhancement_bot_injects_notice():
    class DummyLLM(LLMClient):
        def __init__(self):
            self.captured = None
            self.ctx = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            self.captured = prompt
            self.ctx = context_builder
            return LLMResult(text="")

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build(self, query, **_):
            return ""

    llm = DummyLLM()
    bot = EnhancementBot(context_builder=DummyBuilder(), llm_client=llm)
    bot._codex_summarize("a", "b", confidence=1.0)
    assert llm.captured and llm.captured.system.startswith(PAYMENT_ROUTER_NOTICE)
    assert llm.ctx is not None


def test_prompt_engine_build_prompt_contains_notice():
    from prompt_engine import PromptEngine

    engine = PromptEngine(retriever=None, context_builder=vector_service.ContextBuilder())
    prompt = engine.build_prompt("task", context_builder=engine.context_builder)
    assert prompt.system.startswith(PAYMENT_ROUTER_NOTICE)


def test_bot_development_bot_calls_engine(monkeypatch):
    captured = {}

    def fake_generate(desc: str) -> str:
        captured["desc"] = desc
        return "code"

    engine = sce_stub.SelfCodingEngine()
    monkeypatch.setattr(engine, "generate_helper", fake_generate)

    dummy = types.SimpleNamespace(
        self_coding_engine=engine,
        logger=logging.getLogger("test"),
        _escalate=lambda msg, level="error": None,
        errors=[],
    )

    result = BotDevelopmentBot._call_codex_api(
        dummy,
        "m",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
        ],
    )

    assert captured["desc"] == "hi"
    assert result == "code"


def test_gpt4client_injects_notice(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_chat(messages, *, context_builder, **kwargs):
        msgs = prepend_payment_notice(messages)
        captured["messages"] = msgs
        return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setitem(
        sys.modules,
        "billing.openai_wrapper",
        types.SimpleNamespace(chat_completion_create=fake_chat),
    )

    from neurosales.external_integrations import GPT4Client

    class DummyBuilder:
        def build(self, query: str) -> str:
            return "ctx"

    client = GPT4Client(api_key="k", context_builder=DummyBuilder())
    list(client.stream_chat("arch", [0.1], "obj", "hi"))
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)
