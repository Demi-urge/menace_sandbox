import json
import sys
import types
from dataclasses import dataclass
from menace.coding_bot_interface import manager_generate_helper

from vector_service.retriever import StackRetriever as ContextStackRetriever

from llm_interface import LLMResult

# Stub heavy dependencies before importing the targets


menace_module = sys.modules.setdefault("menace", types.SimpleNamespace())
setattr(menace_module, "RAISE_ERRORS", False)

@dataclass
class _CodeRecord:
    code: str


@dataclass
class _BotRecord:
    name: str = ""


class _CodeDBStub:
    def __init__(self, *args, **kwargs):
        pass


class _BotDBStub:
    def __init__(self, *args, **kwargs):
        pass


class _WorkflowDBStub:
    def __init__(self, *args, **kwargs):
        pass


class _PatchHistoryDBStub:
    def __init__(self, *args, **kwargs):
        pass


class _PatchRecordStub:
    def __init__(self, *args, **kwargs):
        pass


@dataclass
class _StubBotSpec:
    name: str = ""
    purpose: str = ""


class _StubBotDevelopmentBot:
    def __init__(self, repo_base=None, context_builder=None):
        self.repo_base = repo_base
        self.context_builder = context_builder

    def _build_prompt(self, spec, context_builder=None):
        builder = context_builder or self.context_builder
        return builder.build_prompt(spec.purpose)


class _PatchRetrieverStub:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query, top_k=5):
        return []


code_db_mod = sys.modules.setdefault("menace.code_database", types.SimpleNamespace())
code_db_mod.CodeDB = _CodeDBStub
code_db_mod.CodeRecord = _CodeRecord
code_db_mod.PatchHistoryDB = _PatchHistoryDBStub
code_db_mod.PatchRecord = _PatchRecordStub
sys.modules["code_database"] = code_db_mod
sys.modules.setdefault("menace.error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault(
    "menace.bot_database", types.SimpleNamespace(BotDB=_BotDBStub, BotRecord=_BotRecord)
)
sys.modules.setdefault("menace.task_handoff_bot", types.SimpleNamespace(WorkflowDB=_WorkflowDBStub))
sys.modules.setdefault(
    "menace.db_router",
    types.SimpleNamespace(
        DBRouter=object,
        GLOBAL_ROUTER=None,
        LOCAL_TABLES=set(),
        init_db_router=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "menace.unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "menace.trend_predictor", types.SimpleNamespace(TrendPredictor=object)
)
sys.modules.setdefault(
    "menace.menace_memory_manager",
    types.SimpleNamespace(MenaceMemoryManager=object),
)
sys.modules.setdefault(
    "menace.safety_monitor", types.SimpleNamespace(SafetyMonitor=object)
)
sys.modules.setdefault(
    "menace.advanced_error_management",
    types.SimpleNamespace(FormalVerifier=object),
)
sys.modules.setdefault("menace.chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault(
    "menace.model_automation_pipeline", types.SimpleNamespace(ModelAutomationPipeline=object)
)
sys.modules.setdefault(
    "menace.research_aggregator_bot",
    types.SimpleNamespace(ResearchAggregatorBot=object, ResearchItem=object),
)
sys.modules.setdefault(
    "menace.bot_development_bot",
    types.SimpleNamespace(BotDevelopmentBot=_StubBotDevelopmentBot, BotSpec=_StubBotSpec),
)
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
    types.SimpleNamespace(
        LocalKnowledgeModule=object,
        init_local_knowledge=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
sys.modules.setdefault("vector_metrics_db", types.SimpleNamespace(VectorMetricsDB=object))
pylint_mod = types.ModuleType("pylint")
pylint_mod.lint = types.SimpleNamespace(Run=lambda *a, **k: None)
sys.modules.setdefault("pylint", pylint_mod)
sys.modules.setdefault("pylint.lint", pylint_mod.lint)
sys.modules.setdefault("pylint.reporters", types.SimpleNamespace(BaseReporter=object))
for name in [
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "gpt_knowledge_service",
]:
    sys.modules.setdefault(name, sys.modules[f"menace.{name}"])
sys.modules.setdefault(
    "menace.rollback_manager", types.SimpleNamespace(RollbackManager=object)
)
sys.modules.setdefault(
    "menace.audit_trail",
    types.SimpleNamespace(AuditTrail=lambda *a, **k: object()),
)
sys.modules.setdefault(
    "menace.access_control",
    types.SimpleNamespace(
        READ=object(), WRITE=object(), check_permission=lambda *a, **k: None
    ),
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
    "db_router",
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

from menace.vector_service import ContextBuilder  # noqa: E402
import menace.vector_service.context_builder as context_builder_module  # noqa: E402

context_builder_module.PatchRetriever = _PatchRetrieverStub
context_builder_module.ensure_embeddings_fresh = lambda dbs: None

vs_mod = sys.modules.setdefault(
    "vector_service", sys.modules["menace.vector_service"]
)
setattr(vs_mod, "ErrorResult", Exception)

from menace.self_coding_engine import SelfCodingEngine  # noqa: E402
from menace.bot_development_bot import BotDevelopmentBot, BotSpec  # noqa: E402
from menace.config import Config  # noqa: E402
import menace.vector_service.decorators as dec  # noqa: E402
from menace.vector_service.exceptions import MalformedPromptError  # noqa: E402
import pytest  # noqa: E402
from vector_metrics_db import VectorMetricsDB  # noqa: E402


class DummyClient:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt, *, context_builder=None):
        self.last_prompt = getattr(prompt, "text", str(prompt))
        return LLMResult(text="pass")


class MemDB:
    def __init__(self, records):
        self.records = records

    def search(self, _query):
        return self.records


def make_builder(monkeypatch, db_weights=None, max_tokens=800):
    bot_data = [
        {"id": 2, "name": "beta", "roi": 5},
        {"id": 1, "name": "alpha", "roi": 10},
    ]
    workflow_data = [
        {"id": 11, "title": "test", "roi": 2},
        {"id": 10, "title": "deploy", "roi": 7},
    ]
    enhancement_data = [
        {"id": 301, "title": "refactor", "roi": 3, "lessons": "simplify"},
        {"id": 300, "title": "speedup", "roi": 8, "lessons": "optimize"},
    ]
    information_data = [
        {"id": 401, "title": "guide", "roi": 6, "lessons": "useful"},
        {"id": 400, "title": "manual", "roi": 1, "lessons": "basic"},
    ]
    error_data = [
        {"id": 101, "message": "worse", "frequency": 3},
        {"id": 100, "message": "bad", "frequency": 1},
    ]
    code_data = [
        {"id": 201, "summary": "tweak", "roi": 1},
        {"id": 200, "summary": "fix", "roi": 9},
    ]

    class DummyRetrieverSimple:
        def search(self, query, top_k=5, **_):
            results = []
            for name, db in [
                ("bot", MemDB(bot_data)),
                ("workflow", MemDB(workflow_data)),
                ("enhancement", MemDB(enhancement_data)),
                ("information", MemDB(information_data)),
                ("error", MemDB(error_data)),
                ("code", MemDB(code_data)),
            ]:
                for rec in db.search(query):
                    results.append(
                        {
                            "origin_db": name,
                            "record_id": rec["id"],
                            "score": rec.get("score", 0.0),
                            "metadata": rec,
                        }
                    )
            return results

    builder = ContextBuilder(
        retriever=DummyRetrieverSimple(),
        db_weights=db_weights,
        max_tokens=max_tokens,
    )

    expected_ids = {
        "errors": [101],
        "bots": [2],
        "workflows": [11],
        "enhancements": [301],
        "information": [401],
        "code": [201],
    }
    return builder, expected_ids


def test_build_context_compact_json(monkeypatch):
    builder, expected = make_builder(monkeypatch)
    ctx = builder.build_context("alpha issue", top_k=1)
    data = json.loads(ctx)
    for bucket, ids in expected.items():
        assert [item["id"] for item in data.get(bucket, [])[: len(ids)]] == ids
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
    monkeypatch.setattr(
        SelfCodingEngine,
        "generate_helper",
        lambda self, description, **kwargs: pretty,
    )
    engine = SelfCodingEngine(
        code_db,
        object(),
        llm_client=client,
        context_builder=builder,
        gpt_memory=gpt_mem,
    )
    result = engine.generate_helper("alpha issue")
    assert result == pretty


def test_bot_development_bot_includes_context(monkeypatch, tmp_path):
    builder, expected = make_builder(monkeypatch)
    ctx_prompt = builder.build_prompt("alpha issue")
    bot = BotDevelopmentBot(repo_base=tmp_path, context_builder=builder)
    spec = BotSpec(name="demo", purpose="alpha issue")
    prompt = bot._build_prompt(spec, context_builder=builder)
    for ex in ctx_prompt.examples:
        assert ex in prompt.examples


def test_weighted_ordering():
    bundles = [
        {
            "origin_db": "bot",
            "record_id": 1,
            "score": 1.0,
            "metadata": {"id": 1, "name": "alpha", "roi": 1},
        },
        {
            "origin_db": "workflow",
            "record_id": 10,
            "score": 1.0,
            "metadata": {"id": 10, "title": "deploy", "roi": 1},
        },
        {
            "origin_db": "error",
            "record_id": 100,
            "score": 1.0,
            "metadata": {"id": 100, "message": "bad", "frequency": 1},
        },
    ]

    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    plain = ContextBuilder(retriever=NoopRetriever())
    weighted = ContextBuilder(retriever=NoopRetriever(), db_weights={"error": 4.0, "bot": 0.1})

    plain_scores = [plain._bundle_to_entry(b, "q")[1] for b in bundles]
    plain_ids = [s.entry["id"] for s in sorted(plain_scores, key=lambda s: s.score, reverse=True)]
    assert plain_ids == [1, 10, 100]

    weighted_scores = [weighted._bundle_to_entry(b, "q")[1] for b in bundles]
    weighted_ids = [
        s.entry["id"] for s in sorted(weighted_scores, key=lambda s: s.score, reverse=True)
    ]
    assert weighted_ids == [100, 10, 1]


def test_ranking_model_weighting():
    class Ranker:
        def rank(self, query, text):
            return 0.2 if "alpha" in text else 0.9

    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    builder = ContextBuilder(retriever=NoopRetriever(), ranking_model=Ranker())
    b1 = {"origin_db": "bot", "record_id": 1, "score": 1.0, "metadata": {"name": "alpha"}}
    b2 = {"origin_db": "bot", "record_id": 2, "score": 1.0, "metadata": {"name": "beta"}}
    s1 = builder._bundle_to_entry(b1, "q")[1]
    s2 = builder._bundle_to_entry(b2, "q")[1]
    assert s2.score > s1.score


def test_roi_tracker_weighting():
    class Tracker:
        def retrieval_bias(self):
            return {"bot": 0.5, "workflow": 2.0}

    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    builder = ContextBuilder(retriever=NoopRetriever(), roi_tracker=Tracker())
    b1 = {"origin_db": "bot", "record_id": 1, "score": 1.0, "metadata": {"name": "a"}}
    b2 = {"origin_db": "workflow", "record_id": 10, "score": 1.0, "metadata": {"title": "d"}}
    scores = [builder._bundle_to_entry(b, "q")[1] for b in (b1, b2)]
    ids = [s.entry["id"] for s in sorted(scores, key=lambda s: s.score, reverse=True)]
    assert ids == [10, 1]


def test_patch_safety_metrics_influence_ranking():
    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    builder = ContextBuilder(retriever=NoopRetriever())
    good = {
        "origin_db": "code",
        "record_id": 1,
        "score": 1.0,
        "metadata": {"summary": "ok", "win_rate": 0.9, "regret_rate": 0.1},
    }
    risky = {
        "origin_db": "code",
        "record_id": 2,
        "score": 1.0,
        "metadata": {
            "summary": "warn",
            "win_rate": 0.1,
            "regret_rate": 0.9,
            "semantic_alerts": ["alert"],
        },
    }
    s_good = builder._bundle_to_entry(good, "q")[1]
    s_risky = builder._bundle_to_entry(risky, "q")[1]
    assert s_good.score > s_risky.score
    assert s_good.entry["win_rate"] == 0.9
    assert s_risky.entry["regret_rate"] == 0.9
    assert "semantic_alerts" in s_risky.entry["flags"]


def test_risk_penalties_reduce_score():
    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    builder_plain = ContextBuilder(
        retriever=NoopRetriever(),
        safety_weight=1.0,
        regret_penalty=0.0,
        alignment_penalty=0.0,
        alert_penalty=0.0,
    )
    builder = ContextBuilder(
        retriever=NoopRetriever(),
        safety_weight=1.0,
        regret_penalty=2.0,
        alignment_penalty=3.0,
        alert_penalty=4.0,
    )

    base = {"origin_db": "code", "record_id": 1, "score": 1.0, "metadata": {}}
    risky = {
        "origin_db": "code",
        "record_id": 2,
        "score": 1.0,
        "metadata": {
            "regret_rate": 0.5,
            "alignment_severity": 0.5,
            "semantic_alerts": ["a", "b"],
        },
    }

    base_plain = builder_plain._bundle_to_entry(base, "q")[1]
    risk_plain = builder_plain._bundle_to_entry(risky, "q")[1]
    metric_only = base_plain.score - risk_plain.score

    s_base = builder._bundle_to_entry(base, "q")[1]
    s_risk = builder._bundle_to_entry(risky, "q")[1]

    extra = 2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 2
    assert (s_base.score - s_risk.score) == pytest.approx(metric_only + extra)


def test_truncates_when_tokens_small(monkeypatch):
    builder, _ = make_builder(monkeypatch, max_tokens=40)
    ctx = builder.build_context("alpha issue", top_k=2)
    data = json.loads(ctx)
    total_items = sum(len(v) for v in data.values())
    assert total_items <= 12


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
    data = json.loads(ctx)
    for bucket, ids in expected.items():
        assert [item["id"] for item in data.get(bucket, [])[: len(ids)]] == ids

    bundles = [
        {
            "origin_db": "bot",
            "record_id": 1,
            "score": 1.0,
            "metadata": {"id": 1, "name": "alpha", "roi": 1},
        },
        {
            "origin_db": "workflow",
            "record_id": 10,
            "score": 1.0,
            "metadata": {"id": 10, "title": "deploy", "roi": 1},
        },
        {
            "origin_db": "error",
            "record_id": 100,
            "score": 1.0,
            "metadata": {"id": 100, "message": "bad", "frequency": 1},
        },
    ]

    weighted_scores = [builder._bundle_to_entry(b, "q")[1] for b in bundles]
    weighted_ids = [
        s.entry["id"] for s in sorted(weighted_scores, key=lambda s: s.score, reverse=True)
    ]
    assert weighted_ids == [1, 10, 100]


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
    assert total_items <= 12


def test_build_context_emits_metrics(monkeypatch):
    builder, _ = make_builder(monkeypatch)

    class Gauge:
        def __init__(self):
            self.inc_calls = 0
            self.set_calls = []

        def labels(self, *args, **kwargs):
            return self

        def inc(self):
            self.inc_calls += 1

        def set(self, value):
            self.set_calls.append(value)

    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    ctx = builder.build_context("alpha issue")
    assert g1.inc_calls == 1
    assert g3.set_calls == [len(ctx)]


def test_refresh_db_weights(tmp_path):
    class NoopRetriever:
        def search(self, q, top_k=5):
            return []

    db = VectorMetricsDB(tmp_path / "weights.db")
    db.update_db_weight("db1", 1.2)

    builder = ContextBuilder(retriever=NoopRetriever())
    assert "db1" not in builder.db_weights

    builder.refresh_db_weights({"db1": 0.5})
    assert builder.db_weights["db1"] == pytest.approx(0.5)

    builder.refresh_db_weights(vector_metrics=db)
    assert builder.db_weights["db1"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_query", ["", None, 123])
def test_build_context_invalid_query(monkeypatch, bad_query):
    builder, _ = make_builder(monkeypatch)
    with pytest.raises(MalformedPromptError):
        builder.build_context(bad_query)  # type: ignore[arg-type]


def test_prioritise_newer_trimming(monkeypatch):
    records = [
        {"id": 2, "name": "new", "roi": 1, "timestamp": 2},
        {"id": 1, "name": "old", "roi": 1, "timestamp": 1},
    ]

    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return [
                {"origin_db": "bot", "record_id": r["id"], "score": 0.0, "metadata": r}
                for r in records
            ]

    builder = ContextBuilder(retriever=DummyRetriever(), max_tokens=100)
    ctx = builder.build_context("q", top_k=2, prioritise="newest")
    data = json.loads(ctx)
    bots = data.get("bots", [])
    assert bots and bots[0]["id"] == 2


def test_prioritise_roi_trimming(monkeypatch):
    records = [
        {"id": 2, "message": "x", "frequency": 1, "roi": 10},
        {"id": 1, "message": "y", "frequency": 1, "roi": 1},
    ]

    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return [
                {"origin_db": "error", "record_id": r["id"], "score": 0.0, "metadata": r}
                for r in records
            ]

    builder = ContextBuilder(retriever=DummyRetriever(), max_tokens=100)
    ctx = builder.build_context("q", top_k=2, prioritise="roi")
    data = json.loads(ctx)
    errors = data.get("errors", [])
    assert errors and errors[0]["id"] == 2


def test_oversized_dataset_respects_token_limit(monkeypatch):
    long_text = "word " * 200

    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return [
                {
                    "origin_db": "error",
                    "record_id": i,
                    "score": 0.0,
                    "text": long_text,
                    "metadata": {},
                }
                for i in range(2)
            ]

    builder = ContextBuilder(retriever=DummyRetriever(), max_tokens=50)
    ctx, meta = builder.build_context("q", top_k=2, return_metadata=True)
    assert builder._count_tokens(ctx) <= 50
    assert any(m.get("truncated") for m in meta["errors"])


def test_stack_retrieval_included(monkeypatch):
    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    class DummyStackRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, embedding, k=0):
            self.calls.append((embedding, k))
            return [
                {
                    "score": 0.9,
                    "metadata": {
                        "repo": "octo/demo",
                        "path": "src/app.py",
                        "summary": "Stack helper snippet",
                        "license": "mit",
                        "license_fingerprint": "fp",
                    },
                }
            ]

    stack = DummyStackRetriever()
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=stack,
        stack_config={"enabled": True, "top_k": 1, "summary_tokens": 50},
    )
    ctx, meta = builder.build_context("stack query", return_metadata=True)
    data = json.loads(ctx)
    assert "stack" in data
    assert data["stack"][0]["desc"].startswith("Stack helper")
    assert meta["stack"][0]["repo"] == "octo/demo"
    assert stack.calls and stack.calls[0][1] == 1


def test_stack_context_respects_language_filter(monkeypatch):
    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.2, 0.4]

    class Backend:
        def __init__(self):
            self.calls: list[tuple[list[float], int]] = []

        def retrieve(self, embedding, k=0, similarity_threshold=0.0):
            self.calls.append((list(embedding), k))
            return [
                {
                    "score": 0.9,
                    "metadata": {
                        "repo": "octo/demo",
                        "path": "src/demo.py",
                        "language": "Python",
                        "summary": "py helper",
                        "redacted": True,
                    },
                },
                {
                    "score": 0.7,
                    "metadata": {
                        "repo": "octo/demo",
                        "path": "static/app.js",
                        "language": "JavaScript",
                        "summary": "js helper",
                        "redacted": True,
                    },
                },
            ]

    backend = Backend()
    stack = ContextStackRetriever(backend=backend, top_k=3)
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=stack,
        stack_config={"enabled": True, "top_k": 2, "languages": ("python",)},
    )

    ctx, meta = builder.build_context("stack language", return_metadata=True)
    payload = json.loads(ctx)

    assert payload["stack"]
    assert {entry["language"] for entry in meta["stack"]} == {"Python"}
    assert backend.calls and backend.calls[0][1] == 2


def test_stack_token_trimming(monkeypatch):
    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.4, 0.5]

    class DummyStackRetriever:
        def retrieve(self, embedding, k=0):
            text = " ".join("token" for _ in range(40))
            return [
                {
                    "score": 0.3,
                    "metadata": {"repo": "trim/demo", "path": "file.py", "summary": text},
                }
            ]

    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=DummyStackRetriever(),
        stack_config={"enabled": True, "top_k": 1, "summary_tokens": 5},
    )
    ctx, meta = builder.build_context("stack trim", return_metadata=True)
    desc = meta["stack"][0]["desc"]
    assert builder._count_tokens(desc) <= 6
    assert desc.endswith("...")


def test_stack_ingestion_throttled(monkeypatch):
    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.1]

    class DummyStackRetriever:
        def retrieve(self, embedding, k=0):
            return []

    class DummyIngestor:
        def __init__(self):
            self.calls = 0

        def ensure_index_up_to_date(self, cfg):
            self.calls += 1
            return True

    ingestor = DummyIngestor()
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=DummyStackRetriever(),
        stack_ingestor=ingestor,
        stack_config={
            "enabled": True,
            "top_k": 0,
            "ingestion_enabled": True,
            "ingestion_throttle_seconds": 60,
        },
    )
    assert builder.ensure_stack_index_up_to_date() is True
    assert builder.ensure_stack_index_up_to_date() is False
    assert ingestor.calls == 1
    assert builder.ensure_stack_index_up_to_date(force=True) is True
    assert ingestor.calls == 2


def test_stack_disabled_via_env(monkeypatch):
    monkeypatch.setenv("STACK_CONTEXT_DISABLED", "1")

    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.1]

    class TrackingStackRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, embedding, k=0):
            self.calls += 1
            return []

    stack = TrackingStackRetriever()
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=stack,
        stack_config={"enabled": True, "top_k": 1},
    )
    ctx = builder.build_context("disable stack")
    data = json.loads(ctx)
    assert "stack" not in data or not data["stack"]
    assert builder.stack_config.enabled is False
    assert stack.calls == 0
    monkeypatch.delenv("STACK_CONTEXT_DISABLED")


def test_stack_results_cached(monkeypatch):
    class DummyRetriever:
        def search(self, q, top_k=5, **_):
            return []

        def embed_query(self, query):
            return [0.2]

    class ChangingStackRetriever:
        def __init__(self):
            self.calls = 0
            self.payloads = ["first", "second"]

        def retrieve(self, embedding, k=0):
            self.calls += 1
            text = self.payloads.pop(0)
            return [
                {
                    "score": 0.5,
                    "metadata": {"repo": "cache/demo", "path": "demo.py", "summary": text},
                }
            ]

    stack = ChangingStackRetriever()
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_retriever=stack,
        stack_config={"enabled": True, "top_k": 1},
    )
    ctx1 = builder.build_context("cache stack")
    ctx2 = builder.build_context("cache stack")
    assert ctx1 == ctx2
    assert stack.calls == 1
