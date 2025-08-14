import json
import sys
import types

import pytest
from universal_retriever import ResultBundle


# ---------------------------------------------------------------------------
# Lightweight module stubs so imports succeed without heavy dependencies.
sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))


class _DummyCodeDB:
    def __init__(self, *a, **k):
        pass

    def encode_text(self, text):  # pragma: no cover - simple stub
        return [0.0]

    def search_by_vector(self, vector, top_k):  # pragma: no cover - simple stub
        return []

    def get_vector(self, rec_id):  # pragma: no cover - simple stub
        return [0.0]

    def search(self, *a, **k):  # pragma: no cover - simple stub
        return []


_code_stub = types.ModuleType("code_database")
_code_stub.__spec__ = types.SimpleNamespace()
_code_stub.CodeDB = _DummyCodeDB
_code_stub.CodeRecord = object
_code_stub.PatchHistoryDB = _DummyCodeDB
_code_stub.PatchRecord = object
sys.modules.setdefault("code_database", _code_stub)
sys.modules.setdefault("menace.code_database", _code_stub)
sys.modules.setdefault(
    "failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object)
)

# Additional optional dependency stubs for self_coding_engine import.
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
ed = types.ModuleType("ed25519")
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", ed)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", types.ModuleType("serialization")
)
sys.modules.setdefault(
    "env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:")
)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

from menace.context_builder import ContextBuilder


class DummyVecDB:
    """Minimal vector DB used for building the ContextBuilder."""

    def encode_text(self, text):  # pragma: no cover - simple stub
        return [0.0]

    def search_by_vector(self, vector, top_k):  # pragma: no cover - simple stub
        return []

    def get_vector(self, rec_id):  # pragma: no cover - simple stub
        return [0.0]


@pytest.fixture
def builder(monkeypatch):
    mm = types.SimpleNamespace(_summarise_text=lambda text: text[:7])
    db = DummyVecDB()
    cb = ContextBuilder(
        error_db=db,
        bot_db=db,
        workflow_db=db,
        discrepancy_db=db,
        code_db=db,
        memory_manager=mm,
    )

    bundles = [
        ResultBundle("bot", {"id": 1, "name": "AlphaBot", "roi": 10.0}, 0.2, ""),
        ResultBundle("bot", {"id": 2, "name": "BetaBot", "roi": 1.0}, 0.9, ""),
        ResultBundle(
            "workflow", {"id": 3, "title": "AlphaFlow", "roi": 5.0}, 0.1, ""
        ),
        ResultBundle(
            "workflow", {"id": 4, "title": "OtherFlow", "roi": 0.1}, 0.9, ""
        ),
        ResultBundle(
            "error", {"id": 100, "message": "alpha crash", "frequency": 1}, 0.2, ""
        ),
        ResultBundle(
            "discrepancy", {"id": 200, "message": "alpha disc", "severity": 0.9}, 0.2, ""
        ),
        ResultBundle("code", {"id": 42, "summary": "alpha code", "roi": 5.0}, 0.2, ""),
        ResultBundle("code", {"id": 43, "summary": "beta code", "roi": 0.1}, 0.8, ""),
    ]

    monkeypatch.setattr(cb.retriever, "retrieve", lambda q, top_k=15: bundles)
    return cb


def test_build_context_collects_and_prioritises(builder):
    # Generous token budget so all categories are represented
    ctx_json = builder.build_context("alpha", limit_per_db=2, max_tokens=1000)
    ctx = json.loads(ctx_json)

    for key in ("errors", "bots", "workflows", "discrepancies", "code"):
        assert key in ctx and ctx[key]

    # High ROI entries appear first
    assert ctx["bots"][0]["id"] == 1
    assert ctx["workflows"][0]["id"] == 3
    assert ctx["code"][0]["id"] == 42

    # Tight token budget trims lower scoring items
    limited = builder.build_context("alpha", limit_per_db=2, max_tokens=50)
    assert len(limited) // 4 <= 50
    limited_ctx = json.loads(limited)
    assert 43 not in [c["id"] for c in limited_ctx["code"]]


def test_self_coding_engine_prompts_receive_context(builder, tmp_path, monkeypatch):
    import menace.self_coding_engine as sce
    import menace.menace_memory_manager as mm

    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")

    expected = builder.build_context("alpha issue")
    called = {}

    def bc(desc, **kw):
        called["called"] = True
        return expected

    builder.build_context = bc  # type: ignore[assignment]

    engine = sce.SelfCodingEngine(_DummyCodeDB(), mem, context_builder=builder)

    class DummyClient:
        def ask(self, messages, memory_manager=None, tags=None, use_memory=True):
            DummyClient.prompt = messages[0]["content"]
            return {"choices": [{"message": {"content": "def auto_fn():\n    pass"}}]}

    engine.llm_client = DummyClient()
    monkeypatch.setattr(engine, "suggest_snippets", lambda d, limit=3: [])

    engine.generate_helper("alpha issue")
    assert called["called"]

    dumped = json.dumps(expected, indent=2)
    assert "### Retrieval context" in DummyClient.prompt
    assert dumped in DummyClient.prompt

