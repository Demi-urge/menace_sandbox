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
sys.modules.setdefault(
    "menace.gpt_memory",
    types.SimpleNamespace(
        GPTMemoryManager=object,
        STANDARD_TAGS=[],
        INSIGHT="insight",
        _summarise_text=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace.shared_knowledge_module",
    types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None),
)
sys.modules.setdefault(
    "menace.local_knowledge_module",
    types.SimpleNamespace(LocalKnowledgeModule=object, init_local_knowledge=lambda *a, **k: None),
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
for name in [
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "gpt_knowledge_service",
]:
    sys.modules.setdefault(name, sys.modules[f"menace.{name}"])
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

from menace.vector_service import ContextBuilder
from menace.self_coding_engine import SelfCodingEngine
from menace.bot_development_bot import BotDevelopmentBot, BotSpec
from universal_retriever import ResultBundle
from menace.config import Config


class DummyClient:
    def __init__(self):
        self.last_prompt = ""

    def ask(self, messages, **kwargs):
        self.last_prompt = messages[0]["content"]
        return {"choices": [{"message": {"content": "pass"}}]}


class MemDB:
    def __init__(self, records):
        self.records = records

    def search(self, _query):
        return self.records


class DummyRetriever:
    def __init__(self, *, bot_db=None, workflow_db=None, error_db=None):
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.extra = {}

    def register_db(self, name, db, _id_fields):
        if db is not None:
            self.extra[name] = db

    def retrieve(self, query, top_k=5):
        results = []
        for name, db in [
            ("bot", self.bot_db),
            ("workflow", self.workflow_db),
            ("error", self.error_db),
            *self.extra.items(),
        ]:
            if db is None:
                continue
            for rec in db.search(query):
                results.append(ResultBundle(name, rec, rec.get("score", 0.0), ""))
        return results


def make_builder(monkeypatch, db_weights=None, max_tokens=800):
    bot_data = [
        {"id": 2, "name": "beta", "roi": 5},
        {"id": 1, "name": "alpha", "roi": 10},
    ]
    workflow_data = [
        {"id": 11, "title": "test", "roi": 2},
        {"id": 10, "title": "deploy", "roi": 7},
    ]
    error_data = [
        {"id": 101, "message": "worse", "frequency": 3},
        {"id": 100, "message": "bad", "frequency": 1},
    ]
    code_data = [
        {"id": 201, "summary": "tweak", "roi": 1},
        {"id": 200, "summary": "fix", "roi": 9},
    ]
    import menace.vector_service.context_builder as cb_mod
    monkeypatch.setattr(cb_mod, "UniversalRetriever", DummyRetriever)

    builder = ContextBuilder(
        bot_db=MemDB(bot_data),
        workflow_db=MemDB(workflow_data),
        error_db=MemDB(error_data),
        code_db=MemDB(code_data),
        db_weights=db_weights,
        max_tokens=max_tokens,
    )

    expected = {
        "errors": [{"id": 100, "desc": "bad", "metric": 0.5}],
        "bots": [
            {"id": 1, "name": "alpha", "desc": "alpha", "metric": 10.0}
        ],
        "workflows": [
            {"id": 10, "title": "deploy", "desc": "deploy", "metric": 7.0}
        ],
        "code": [{"id": 200, "desc": "fix", "metric": 9.0}],
        "discrepancies": [],
    }
    expected_str = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    return builder, expected_str


def test_build_context_compact_json(monkeypatch):
    builder, expected = make_builder(monkeypatch)
    ctx = builder.build_context("alpha issue", top_k=1)
    assert ctx == expected
    assert ": " not in ctx
    assert ", " not in ctx


def test_self_coding_engine_includes_context(monkeypatch):
    builder, expected = make_builder(monkeypatch)
    # Pre-compute context for pretty representation
    pretty = json.dumps(json.loads(builder.build_context("alpha issue")), indent=2)
    code_db = types.SimpleNamespace(search=lambda q: [{"code": "print('x')"}])
    gpt_mem = types.SimpleNamespace(
        search_context=lambda *a, **k: [], log_interaction=lambda *a, **k: None
    )
    client = DummyClient()
    engine = SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        context_builder=builder,
        gpt_memory=gpt_mem,
    )
    engine.generate_helper("alpha issue")
    assert "### Retrieval context" in client.last_prompt
    assert pretty in client.last_prompt


def test_bot_development_bot_includes_context(monkeypatch, tmp_path):
    builder, expected = make_builder(monkeypatch)
    ctx = builder.build_context("alpha issue")
    bot = BotDevelopmentBot(repo_base=tmp_path, context_builder=builder)
    spec = BotSpec(name="demo", purpose="alpha issue")
    prompt = bot._build_prompt(spec)
    assert "\n\nContext:\n" in prompt
    assert ctx in prompt


def test_weighted_ordering(monkeypatch):
    import menace.vector_service.context_builder as cb_mod

    monkeypatch.setattr(cb_mod, "UniversalRetriever", DummyRetriever)

    bundles = [
        ResultBundle("bot", {"id": 1, "name": "alpha", "roi": 1}, 1.0, ""),
        ResultBundle("workflow", {"id": 10, "title": "deploy", "roi": 1}, 1.0, ""),
        ResultBundle("error", {"id": 100, "message": "bad", "frequency": 1}, 1.0, ""),
    ]

    plain = ContextBuilder()
    weighted = ContextBuilder(db_weights={"error": 4.0, "bot": 0.1})

    plain_scores = [plain._bundle_to_entry(b)[1] for b in bundles]
    plain_ids = [s.entry["id"] for s in sorted(plain_scores, key=lambda s: s.score, reverse=True)]
    assert plain_ids == [1, 10, 100]

    weighted_scores = [weighted._bundle_to_entry(b)[1] for b in bundles]
    weighted_ids = [
        s.entry["id"] for s in sorted(weighted_scores, key=lambda s: s.score, reverse=True)
    ]
    assert weighted_ids == [100, 10, 1]


def test_truncates_when_tokens_small(monkeypatch):
    builder, _ = make_builder(monkeypatch, max_tokens=40)
    ctx = builder.build_context("alpha issue", top_k=2)
    data = json.loads(ctx)
    total_items = sum(len(v) for v in data.values())
    assert total_items < 8
    assert len(ctx) // 4 <= builder.max_tokens


def test_builder_from_config(monkeypatch):
    import menace.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "UnifiedConfigStore", None)
    monkeypatch.setattr(cfg_mod, "_CONFIG_STORE", None)

    overrides = {
        "context_builder": {"db_weights": {"error": 4.0, "bot": 0.1}, "max_tokens": 400}
    }
    cfg = Config.from_overrides(overrides)

    builder, expected = make_builder(
        monkeypatch,
        db_weights=cfg.context_builder.db_weights,
        max_tokens=cfg.context_builder.max_tokens,
    )

    assert builder.db_weights == cfg.context_builder.db_weights
    assert builder.max_tokens == cfg.context_builder.max_tokens

    ctx = builder.build_context("alpha issue", top_k=1)
    assert ctx == expected

    bundles = [
        ResultBundle("bot", {"id": 1, "name": "alpha", "roi": 1}, 1.0, ""),
        ResultBundle("workflow", {"id": 10, "title": "deploy", "roi": 1}, 1.0, ""),
        ResultBundle("error", {"id": 100, "message": "bad", "frequency": 1}, 1.0, ""),
    ]

    weighted_scores = [builder._bundle_to_entry(b)[1] for b in bundles]
    weighted_ids = [
        s.entry["id"] for s in sorted(weighted_scores, key=lambda s: s.score, reverse=True)
    ]
    assert weighted_ids == [100, 10, 1]


def test_config_respects_max_tokens(monkeypatch):
    import menace.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "UnifiedConfigStore", None)
    monkeypatch.setattr(cfg_mod, "_CONFIG_STORE", None)

    overrides = {"context_builder": {"max_tokens": 40, "db_weights": {"error": 2.0}}}
    cfg = Config.from_overrides(overrides)

    builder, _ = make_builder(
        monkeypatch,
        db_weights=cfg.context_builder.db_weights,
        max_tokens=cfg.context_builder.max_tokens,
    )

    ctx = builder.build_context("alpha issue", top_k=2)
    data = json.loads(ctx)
    total_items = sum(len(v) for v in data.values())
    assert builder.db_weights == cfg.context_builder.db_weights
    assert builder.max_tokens == cfg.context_builder.max_tokens
    assert total_items < 8
    assert len(ctx) // 4 <= cfg.context_builder.max_tokens
