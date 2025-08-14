import json
import sys
import types
from dataclasses import dataclass

# Stub heavy dependencies before importing the targets

@dataclass
class _CodeRecord:
    code: str

@dataclass
class _BotRecord:
    name: str = ""

sys.modules.setdefault(
    "menace.code_database",
    types.SimpleNamespace(
        CodeDB=object, CodeRecord=_CodeRecord, PatchHistoryDB=object, PatchRecord=object
    ),
)
sys.modules.setdefault("menace.error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault(
    "menace.bot_database", types.SimpleNamespace(BotDB=object, BotRecord=_BotRecord)
)
sys.modules.setdefault("menace.task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("menace.database_router", types.SimpleNamespace(DatabaseRouter=object))
sys.modules.setdefault("menace.unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
sys.modules.setdefault("menace.trend_predictor", types.SimpleNamespace(TrendPredictor=object))
sys.modules.setdefault("menace.menace_memory_manager", types.SimpleNamespace(MenaceMemoryManager=object))
sys.modules.setdefault("menace.safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
sys.modules.setdefault("menace.advanced_error_management", types.SimpleNamespace(FormalVerifier=object))
sys.modules.setdefault("menace.chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault("menace.gpt_memory", types.SimpleNamespace(GPTMemoryManager=object, GPTMemory=object))
sys.modules.setdefault("menace.rollback_manager", types.SimpleNamespace(RollbackManager=object))
sys.modules.setdefault("menace.audit_trail", types.SimpleNamespace(AuditTrail=lambda *a, **k: object()))
sys.modules.setdefault(
    "menace.access_control",
    types.SimpleNamespace(READ=object(), WRITE=object(), check_permission=lambda *a, **k: None),
)
sys.modules.setdefault(
    "menace.patch_suggestion_db",
    types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object),
)

# Provide top-level aliases for modules imported without package prefix
for name in [
    "code_database",
    "error_bot",
    "bot_database",
    "task_handoff_bot",
    "database_router",
    "unified_event_bus",
    "trend_predictor",
    "menace_memory_manager",
    "safety_monitor",
    "advanced_error_management",
    "chatgpt_idea_bot",
    "gpt_memory",
    "rollback_manager",
    "audit_trail",
    "access_control",
    "patch_suggestion_db",
]:
    sys.modules.setdefault(name, sys.modules[f"menace.{name}"])

from menace.self_coding_engine import SelfCodingEngine
from menace.bot_development_bot import BotDevelopmentBot, BotSpec


class DummyClient:
    def __init__(self):
        self.last_prompt = ""

    def ask(self, messages, **kwargs):
        self.last_prompt = messages[0]["content"]
        return {"choices": [{"message": {"content": "pass"}}]}


def test_self_coding_engine_includes_context(monkeypatch):
    ctx = {"bots": [{"id": 1, "summary": "alpha bot"}]}
    builder = types.SimpleNamespace(build_context=lambda q: ctx)
    code_db = types.SimpleNamespace(search=lambda q: [{"code": "print('x')"}])
    memory_mgr = object()
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [], log_interaction=lambda *a, **k: None
    )
    client = DummyClient()
    engine = SelfCodingEngine(
        code_db,
        memory_mgr,
        llm_client=client,
        context_builder=builder,
        gpt_memory=gpt_mem,
    )
    engine.generate_helper("alpha task")
    assert "### Retrieval context" in client.last_prompt
    assert json.dumps(ctx, indent=2) in client.last_prompt


def test_bot_development_bot_includes_context(tmp_path):
    ctx = {"errors": [{"id": 2, "summary": "failure"}]}
    builder = types.SimpleNamespace(build_context=lambda q: ctx)
    bot = BotDevelopmentBot(repo_base=tmp_path, context_builder=builder)
    spec = BotSpec(name="demo", purpose="test")
    prompt = bot._build_prompt(spec)
    assert "\n\nContext:\n" in prompt
    assert json.dumps(ctx, indent=2) in prompt
