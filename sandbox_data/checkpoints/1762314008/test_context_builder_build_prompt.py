import types
import sys
from pathlib import Path

# Stub heavy dependencies before importing the target module
sys.modules.setdefault("retrieval_cache", types.SimpleNamespace(RetrievalCache=object))
sys.modules.setdefault(
    "db_router",
    types.SimpleNamespace(DBRouter=object, GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None),
)
sys.modules.setdefault(
    "dynamic_path_router", types.SimpleNamespace(resolve_path=lambda *a, **k: Path("."))
)
sys.modules.setdefault("filelock", types.SimpleNamespace(FileLock=lambda *a, **k: None))
sys.modules.setdefault(
    "redaction_utils",
    types.SimpleNamespace(redact_text=lambda x: x, redact_dict=lambda x: x),
)
sys.modules.setdefault("patch_safety", types.SimpleNamespace(PatchSafety=object))
sys.modules.setdefault(
    "vector_service.ranking_utils", types.SimpleNamespace(rank_patches=lambda *a, **k: ([], 0.0))
)
sys.modules.setdefault(
    "vector_service.embedding_backfill",
    types.SimpleNamespace(
        ensure_embeddings_fresh=lambda *a, **k: None,
        StaleEmbeddingsError=Exception,
        EmbeddingBackfill=object,
        schedule_backfill=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "vector_service.patch_logger", types.SimpleNamespace(_VECTOR_RISK=None)
)
sys.modules.setdefault(
    "compliance.license_fingerprint", types.SimpleNamespace(DENYLIST={})
)
sys.modules.setdefault(
    "vector_service.retriever",
    types.SimpleNamespace(Retriever=object, PatchRetriever=object, FallbackResult=list),
)
sys.modules.setdefault(
    "config",
    types.SimpleNamespace(
        ContextBuilderConfig=lambda: types.SimpleNamespace(
            ranking_weight=1.0,
            roi_weight=1.0,
            recency_weight=1.0,
            safety_weight=1.0,
            max_tokens=800,
            regret_penalty=1.0,
            alignment_penalty=1.0,
            alert_penalty=1.0,
            risk_penalty=1.0,
            roi_tag_penalties={},
            enhancement_weight=1.0,
            max_alignment_severity=1.0,
            max_alerts=5,
            license_denylist=set(),
            precise_token_count=False,
            max_diff_lines=200,
            similarity_metric="cosine",
            embedding_check_interval=0,
            prompt_score_weight=1.0,
            prompt_max_tokens=800,
        )
    ),
)
sys.modules.setdefault(
    "config",
    types.SimpleNamespace(
        ContextBuilderConfig=lambda: types.SimpleNamespace(
            ranking_weight=1.0,
            roi_weight=1.0,
            recency_weight=1.0,
            safety_weight=1.0,
            max_tokens=800,
            regret_penalty=1.0,
            alignment_penalty=1.0,
            alert_penalty=1.0,
            risk_penalty=1.0,
            roi_tag_penalties={},
            enhancement_weight=1.0,
            max_alignment_severity=1.0,
            max_alerts=5,
            license_denylist=set(),
            precise_token_count=False,
            max_diff_lines=200,
            similarity_metric="cosine",
            embedding_check_interval=0,
            prompt_score_weight=1.0,
            prompt_max_tokens=800,
        )
    ),
)

from vector_service.context_builder import ContextBuilder


def _make_builder() -> ContextBuilder:
    b = ContextBuilder.__new__(ContextBuilder)
    b._count_tokens = lambda text: len(str(text).split())  # type: ignore[attr-defined]
    b.prompt_max_tokens = 100
    b.prompt_score_weight = 1.0
    b.roi_weight = 1.0
    return b


def test_build_prompt_enrichment():
    builder = _make_builder()
    calls: list[str] = []

    def fake_build_context(self, query, **kwargs):
        calls.append(query)
        meta = {"bots": [{"desc": query, "score": 1.0, "roi": 1}]}
        return "{}", "sess", [], meta

    builder.build_context = types.MethodType(fake_build_context, builder)
    prompt = builder.build_prompt("alpha", latent_queries=["beta"])
    assert calls == ["alpha", "beta"]
    assert set(prompt.examples) == {"alpha", "beta"}


def test_build_prompt_dedup_priority():
    builder = _make_builder()

    def fake_build_context(self, query, **kwargs):
        meta = {
            "bots": [{"desc": "dup", "score": 0.1, "roi": 0}],
            "code": [{"desc": "dup", "score": 0.9, "roi": 0}],
            "info": [{"desc": "other", "score": 0.2, "roi": 0}],
        }
        return "{}", "sess", [], meta

    builder.build_context = types.MethodType(fake_build_context, builder)
    prompt = builder.build_prompt("task")
    assert prompt.examples[0] == "dup"
    assert prompt.examples[1] == "other"
    assert prompt.examples.count("dup") == 1


def test_build_prompt_token_budget():
    builder = _make_builder()
    builder.prompt_max_tokens = 1

    def fake_build_context(self, query, **kwargs):
        meta = {"bots": [{"desc": "one", "score": 1.0}, {"desc": "two", "score": 0.5}]}
        return "{}", "sess", [], meta

    builder.build_context = types.MethodType(fake_build_context, builder)
    prompt = builder.build_prompt("intent")
    assert prompt.examples == ["one"]

