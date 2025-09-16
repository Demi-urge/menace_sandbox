import json
import sys
import types
from dataclasses import dataclass
from menace.coding_bot_interface import manager_generate_helper

from llm_interface import LLMResult

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

    expected = {
        "errors": [{"id": 100, "desc": "bad", "metric": 0.5}],
        "bots": [
            {"id": 1, "name": "alpha", "desc": "alpha", "metric": 10.0}
        ],
        "workflows": [
            {"id": 10, "title": "deploy", "desc": "deploy", "metric": 7.0}
        ],
        "enhancements": [
            {
                "id": 300,
                "title": "speedup",
                "lessons": "optimize",
                "desc": "speedup",
                "metric": 8.0,
            }
        ],
        "information": [
            {
                "id": 401,
                "title": "guide",
                "lessons": "useful",
                "desc": "guide",
                "metric": 6.0,
            }
        ],
        "code": [{"id": 200, "desc": "fix", "metric": 9.0}],
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
    manager = types.SimpleNamespace(engine=engine)
    manager_generate_helper(manager, "alpha issue", context_builder=builder)
    assert "### Retrieval context" in client.last_prompt
    assert pretty in client.last_prompt


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
    assert ctx == expected

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

    builder = ContextBuilder(retriever=DummyRetriever(), max_tokens=10)
    ctx = builder.build_context("q", top_k=2, prioritise="newest")
    data = json.loads(ctx)
    assert data["bots"][0]["id"] == 2


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

    builder = ContextBuilder(retriever=DummyRetriever(), max_tokens=10)
    ctx = builder.build_context("q", top_k=2, prioritise="roi")
    data = json.loads(ctx)
    assert data["errors"][0]["id"] == 2


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
