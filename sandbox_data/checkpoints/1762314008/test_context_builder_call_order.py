import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

if "menace.self_coding_engine" not in sys.modules:
    sce_stub = types.ModuleType("menace.self_coding_engine")

    class _ManagerContext:
        def set(self, manager):  # pragma: no cover - stub
            return None

        def reset(self, token):  # pragma: no cover - stub
            return None

    sce_stub.MANAGER_CONTEXT = _ManagerContext()
    sys.modules["menace.self_coding_engine"] = sce_stub
    sys.modules.setdefault("menace_sandbox.self_coding_engine", sce_stub)

coding_stub = types.ModuleType("coding_bot_interface")


def _noop_self_coding_managed(*args, **kwargs):
    def decorator(cls):
        return cls

    return decorator


coding_stub.self_coding_managed = _noop_self_coding_managed
sys.modules["coding_bot_interface"] = coding_stub
sys.modules.setdefault("menace.coding_bot_interface", coding_stub)
sys.modules.setdefault("menace_sandbox.coding_bot_interface", coding_stub)

menace_pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
setattr(menace_pkg, "RAISE_ERRORS", False)

neurosales_stub = types.ModuleType("neurosales")


class _Entry(types.SimpleNamespace):
    pass


neurosales_stub.MessageEntry = _Entry
neurosales_stub.CTAChain = _Entry
neurosales_stub.add_message = lambda *a, **k: None
neurosales_stub.get_recent_messages = lambda *a, **k: []
neurosales_stub.push_chain = lambda *a, **k: None
neurosales_stub.peek_chain = lambda: None
sys.modules["neurosales"] = neurosales_stub
sys.modules.setdefault("menace.neurosales", neurosales_stub)
sys.modules.setdefault("menace_sandbox.neurosales", neurosales_stub)

bot_registry_stub = types.ModuleType("bot_registry")


def _register_bot(*a, **k):  # pragma: no cover - stub
    return None


class _BotRegistry:
    register_bot = staticmethod(_register_bot)


bot_registry_stub.BotRegistry = _BotRegistry
sys.modules["bot_registry"] = bot_registry_stub
sys.modules.setdefault("menace.bot_registry", bot_registry_stub)
sys.modules.setdefault("menace_sandbox.bot_registry", bot_registry_stub)

data_bot_stub = types.ModuleType("data_bot")


class _DataBot:
    def __init__(self, *a, **k):  # pragma: no cover - stub
        pass


data_bot_stub.DataBot = _DataBot
sys.modules["data_bot"] = data_bot_stub
sys.modules.setdefault("menace.data_bot", data_bot_stub)
sys.modules.setdefault("menace_sandbox.data_bot", data_bot_stub)

report_stub = types.ModuleType("report_generation_bot")


class _ReportGenerationBot:
    def __init__(self, *a, **k):  # pragma: no cover - stub
        pass


class _ReportOptions:  # pragma: no cover - stub
    pass


report_stub.ReportGenerationBot = _ReportGenerationBot
report_stub.ReportOptions = _ReportOptions
sys.modules["report_generation_bot"] = report_stub
sys.modules.setdefault("menace.report_generation_bot", report_stub)
sys.modules.setdefault("menace_sandbox.report_generation_bot", report_stub)

db_mgmt_stub = types.ModuleType("database_management_bot")


class _DatabaseManagementBot:
    def __init__(self, *a, **k):  # pragma: no cover - stub
        pass

    def ingest_idea(self, *a, **k):  # pragma: no cover - stub
        return None


db_mgmt_stub.DatabaseManagementBot = _DatabaseManagementBot
sys.modules["database_management_bot"] = db_mgmt_stub
sys.modules.setdefault("menace.database_management_bot", db_mgmt_stub)
sys.modules.setdefault("menace_sandbox.database_management_bot", db_mgmt_stub)

log_tags_stub = types.ModuleType("log_tags")
log_tags_stub.FEEDBACK = "feedback"
log_tags_stub.IMPROVEMENT_PATH = "improvement"
log_tags_stub.ERROR_FIX = "error"
log_tags_stub.INSIGHT = "insight"
sys.modules["log_tags"] = log_tags_stub
sys.modules.setdefault("menace.log_tags", log_tags_stub)
sys.modules.setdefault("menace_sandbox.log_tags", log_tags_stub)

shared_memory_stub = types.ModuleType("shared_gpt_memory")
shared_memory_stub.GPT_MEMORY_MANAGER = object()
sys.modules["shared_gpt_memory"] = shared_memory_stub
sys.modules.setdefault("menace.shared_gpt_memory", shared_memory_stub)
sys.modules.setdefault("menace_sandbox.shared_gpt_memory", shared_memory_stub)

gpt_memory_stub = types.ModuleType("gpt_memory_interface")


class _GPTMemoryInterface:  # pragma: no cover - stub
    pass


gpt_memory_stub.GPTMemoryInterface = _GPTMemoryInterface
sys.modules["gpt_memory_interface"] = gpt_memory_stub

# Stubs for QueryBot dependencies -------------------------------------------------------
database_manager_stub = types.ModuleType("database_manager")
database_manager_stub.DB_PATH = "db"
database_manager_stub.search_models = lambda *a, **k: []
sys.modules["database_manager"] = database_manager_stub
sys.modules.setdefault("menace.database_manager", database_manager_stub)
sys.modules.setdefault("menace_sandbox.database_manager", database_manager_stub)

local_knowledge_stub = types.ModuleType("local_knowledge_module")


class _LocalKnowledgeModule:
    def __init__(self, *a, **k):  # pragma: no cover - stub
        pass


local_knowledge_stub.LocalKnowledgeModule = _LocalKnowledgeModule
sys.modules["local_knowledge_module"] = local_knowledge_stub
sys.modules.setdefault("menace.local_knowledge_module", local_knowledge_stub)
sys.modules.setdefault("menace_sandbox.local_knowledge_module", local_knowledge_stub)

run_auto_stub = types.ModuleType("run_autonomous")
run_auto_stub.LOCAL_KNOWLEDGE_MODULE = None
sys.modules["run_autonomous"] = run_auto_stub
sys.modules.setdefault("menace.run_autonomous", run_auto_stub)
sys.modules.setdefault("menace_sandbox.run_autonomous", run_auto_stub)

sandbox_runner_stub = types.ModuleType("sandbox_runner")
sandbox_runner_stub.LOCAL_KNOWLEDGE_MODULE = None
sys.modules["sandbox_runner"] = sandbox_runner_stub
sys.modules.setdefault("menace.sandbox_runner", sandbox_runner_stub)
sys.modules.setdefault("menace_sandbox.sandbox_runner", sandbox_runner_stub)

# Load actual chatgpt_idea_bot module with dependencies stubbed
chatgpt_path = Path(__file__).resolve().parents[1] / "chatgpt_idea_bot.py"
spec = importlib.util.spec_from_file_location("menace.chatgpt_idea_bot", chatgpt_path)
cib_module = importlib.util.module_from_spec(spec)
sys.modules["menace.chatgpt_idea_bot"] = cib_module
sys.modules["chatgpt_idea_bot"] = cib_module
spec.loader.exec_module(cib_module)

import menace.conversation_manager_bot as cmb  # noqa: E402
import menace.query_bot as qb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402
from prompt_types import Prompt  # noqa: E402


class RecordingBuilder:
    def refresh_db_weights(self) -> None:
        pass

    def build_prompt(self, query: str, **kwargs):  # pragma: no cover - replaced in tests
        return Prompt(user=query)


def test_conversation_manager_uses_builder_before_generate(monkeypatch):
    builder = RecordingBuilder()
    call_order: list[tuple[str, str]] = []

    def record_build(query: str, **kwargs):
        call_order.append(("build_prompt", query))
        return Prompt(user=query)

    monkeypatch.setattr(builder, "build_prompt", record_build)
    monkeypatch.setattr(cmb, "mq_add_message", lambda *a, **k: None)
    monkeypatch.setattr(cmb, "get_recent_messages", lambda *a, **k: [])
    monkeypatch.setattr(cmb, "push_chain", lambda *a, **k: None)
    monkeypatch.setattr(cmb, "peek_chain", lambda: None)

    class DummyClient:
        def __init__(self) -> None:
            self.context_builder = builder
            self.gpt_memory = None

        def generate(self, prompt_obj, *, context_builder, **kwargs):
            call_order.append(("generate", prompt_obj.user))
            assert context_builder is builder
            return SimpleNamespace(text="ok", raw={})

    bot = cmb.ConversationManagerBot(DummyClient())
    result = bot._chatgpt("hello")

    assert result == "ok"
    assert [name for name, _ in call_order] == ["build_prompt", "generate"]


def test_query_bot_process_calls_builder_before_generate(monkeypatch):
    builder = RecordingBuilder()
    call_order: list[tuple[str, str]] = []

    def record_build(query: str, **kwargs):
        call_order.append(("build_prompt", query))
        return Prompt(user=f"prompt:{query}")

    monkeypatch.setattr(builder, "build_prompt", record_build)

    class DummyClient:
        def __init__(self) -> None:
            self.context_builder = builder
            self.gpt_memory = None

        def generate(self, prompt_obj, *, context_builder, **kwargs):
            call_order.append(("generate", prompt_obj.user))
            assert context_builder is builder
            return SimpleNamespace(text="answer", raw={})

    client = DummyClient()
    fetcher = qb.DataFetcher(data={"foo": {"value": 1}})
    store = qb.ContextStore()
    bot = qb.QueryBot(
        client=client,
        fetcher=fetcher,
        store=store,
        nlu=qb.SimpleNLU(),
        knowledge=SimpleNamespace(),
        context_builder=builder,
    )

    result = bot.process("get foo", "cid-1")

    assert result.text == "answer"
    assert [name for name, _ in call_order] == ["build_prompt", "generate"]


def test_follow_up_builds_prompt_before_generate(monkeypatch):
    builder = RecordingBuilder()
    call_order: list[tuple[str, str]] = []

    def record_build(prompt: str, **kwargs):
        call_order.append(("build_prompt", prompt))
        return Prompt(user=prompt)

    monkeypatch.setattr(builder, "build_prompt", record_build)

    class DummyClient:
        def __init__(self) -> None:
            self.context_builder = builder
            self.gpt_memory = None

        def generate(self, prompt_obj, *, context_builder, **kwargs):
            call_order.append(("generate", prompt_obj.user))
            assert context_builder is builder
            return SimpleNamespace(text="insight", raw={})

    idea = cib.Idea(name="Idea", description="Desc")
    output = cib.follow_up(DummyClient(), idea, context_builder=builder)

    assert output == "insight"
    assert idea.insight == "insight"
    assert [name for name, _ in call_order] == ["build_prompt", "generate"]
