import types
import sys
import logging
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
sys.modules.setdefault("patch_safety", types.SimpleNamespace(PatchSafety=type("PS", (), {})))
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
    types.SimpleNamespace(
        Retriever=type("R", (), {"__init__": lambda self, *a, **k: None}),
        PatchRetriever=type("PR", (), {"__init__": lambda self, *a, **k: None}),
        FallbackResult=list,
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

from vector_service.context_builder import ContextBuilder, logger


class FailingPatchRetriever:
    roi_tag_weights = {}

    def __setattr__(self, name, value):
        raise RuntimeError("boom")


def test_patch_retriever_config_logged(monkeypatch):
    called = {}

    def fake_exception(msg):
        called["msg"] = msg

    monkeypatch.setattr(logger, "exception", fake_exception)
    ContextBuilder(patch_retriever=FailingPatchRetriever())
    assert called.get("msg") == "patch_retriever configuration failed"
