import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Provide lightweight stubs for heavy dependencies so the real modules can be
# imported without initialising databases or accessing configuration files.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault(
    "retrieval_cache", types.SimpleNamespace(RetrievalCache=type("RetrievalCache", (), {}))
)
sys.modules.setdefault(
    "vector_service.retriever",
    types.SimpleNamespace(
        Retriever=type("Retriever", (), {"__init__": lambda self, *a, **k: None}),
        PatchRetriever=type(
            "PatchRetriever", (), {"__init__": lambda self, *a, **k: None}
        ),
        FallbackResult=type("FallbackResult", (), {}),
    ),
)
sys.modules.setdefault("vector_service.patch_logger", types.SimpleNamespace(_VECTOR_RISK=0))
sys.modules.setdefault("vector_metrics_db", types.SimpleNamespace(VectorMetricsDB=None))
sys.modules.setdefault(
    "compliance.license_fingerprint", types.SimpleNamespace(DENYLIST={})
)


class ContextBuilderConfig:
    def __init__(self):
        self.ranking_weight = 1.0
        self.roi_weight = 1.0
        self.recency_weight = 1.0
        self.safety_weight = 1.0
        self.max_tokens = 1000
        self.regret_penalty = 0.0
        self.alignment_penalty = 0.0
        self.alert_penalty = 0.0
        self.risk_penalty = 1.0
        self.roi_tag_penalties = {}
        self.enhancement_weight = 1.0
        self.max_alignment_severity = 1.0
        self.max_alerts = 5
        self.license_denylist = set()
        self.precise_token_count = False
        self.max_diff_lines = 200
        self.similarity_metric = "cosine"


sys.modules.setdefault("config", types.SimpleNamespace(ContextBuilderConfig=ContextBuilderConfig))

# Create a lightweight package so submodules can be imported without executing
# the actual ``vector_service`` package ``__init__``.
pkg = types.ModuleType("vector_service")
pkg.__path__ = [str(ROOT / "vector_service")]
sys.modules.setdefault("vector_service", pkg)

from vector_service.context_builder_utils import get_default_context_builder
from vector_service.context_builder import ContextBuilder


def test_get_default_context_builder_returns_builder():
    dummy_retriever = types.SimpleNamespace()
    builder = get_default_context_builder(
        retriever=dummy_retriever, precise_token_count=False
    )
    assert isinstance(builder, ContextBuilder)
    builder.refresh_db_weights({})


def test_kwargs_forwarded_to_context_builder():
    dummy_retriever = types.SimpleNamespace()
    builder = get_default_context_builder(
        retriever=dummy_retriever, max_tokens=1, precise_token_count=False
    )
    assert builder.max_tokens == 1
