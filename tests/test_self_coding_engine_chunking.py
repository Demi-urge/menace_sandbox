from __future__ import annotations

from pathlib import Path
import sys
import types
import importlib.util
import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs so the self_coding_engine module can be imported without
# pulling in heavy optional dependencies. These are intentionally minimal and
# mirror the approach used in other tests.

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("menace_sandbox")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("menace_sandbox", package)


def _setmod(name: str, module: object) -> None:
    sys.modules.setdefault(name, module)
    sys.modules.setdefault(f"menace_sandbox.{name}", module)


vec_mod = types.ModuleType("vector_service")


class _VSError(Exception):
    pass


vec_mod.CognitionLayer = object
vec_mod.PatchLogger = object
vec_mod.VectorServiceError = _VSError
vec_mod.SharedVectorService = object
vec_mod.ContextBuilder = object
_setmod("vector_service", vec_mod)
_setmod("vector_service.retriever", types.ModuleType("vector_service.retriever"))
_setmod("vector_service.decorators", types.ModuleType("vector_service.decorators"))
_setmod(
    "vector_service.text_preprocessor",
    types.SimpleNamespace(
        PreprocessingConfig=object,
        get_config=lambda *a, **k: None,
        generalise=lambda *a, **k: "",
    ),
)
_setmod(
    "vector_service.embed_utils",
    types.SimpleNamespace(get_text_embeddings=lambda *a, **k: [], EMBED_DIM=0),
)

builder = types.SimpleNamespace(
    build_context=lambda *a, **k: {},
    refresh_db_weights=lambda *a, **k: None,
)

code_db_mod = types.ModuleType("code_database")
code_db_mod.CodeDB = object
code_db_mod.CodeRecord = object
code_db_mod.PatchHistoryDB = object
code_db_mod.PatchRecord = object
_setmod("code_database", code_db_mod)

_setmod("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
_setmod("trend_predictor", types.SimpleNamespace(TrendPredictor=object))
_setmod("gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object))
_setmod("safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
_setmod("advanced_error_management", types.SimpleNamespace(FormalVerifier=object))
_setmod("shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None))
_setmod("gpt_memory", types.SimpleNamespace(GPTMemoryManager=object))
_setmod("menace_sanity_layer", types.SimpleNamespace(fetch_recent_billing_issues=lambda: []))
log_tags_mod = types.SimpleNamespace(
    FEEDBACK="feedback",
    ERROR_FIX="error_fix",
    IMPROVEMENT_PATH="improvement_path",
    INSIGHT="insight",
)
_setmod("log_tags", log_tags_mod)
_setmod("gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object))
know_mod = types.ModuleType("knowledge_retriever")
know_mod.get_feedback = lambda *a, **k: []
know_mod.get_error_fixes = lambda *a, **k: []
know_mod.recent_feedback = lambda *a, **k: None
know_mod.recent_error_fix = lambda *a, **k: None
know_mod.recent_improvement_path = lambda *a, **k: None
_setmod("knowledge_retriever", know_mod)
_setmod("rollback_manager", types.SimpleNamespace(RollbackManager=object))
audit_mod = types.ModuleType("audit_trail")
audit_mod.AuditTrail = lambda *a, **k: types.SimpleNamespace(
    record=lambda self, payload: None
)
_setmod("audit_trail", audit_mod)
access_mod = types.SimpleNamespace(
    READ="r",
    WRITE="w",
    check_permission=lambda *a, **k: None,
)
_setmod("access_control", access_mod)
_setmod(
    "patch_suggestion_db",
    types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object),
)
_setmod(
    "sandbox_runner.workflow_sandbox_runner",
    types.SimpleNamespace(WorkflowSandboxRunner=object),
)
_setmod(
    "sandbox_runner.test_harness",
    types.SimpleNamespace(
        run_tests=lambda *a, **k: None,
        TestHarnessResult=types.SimpleNamespace(success=False, stdout=""),
    ),
)

_setmod(
    "sandbox_settings",
    types.SimpleNamespace(
        SandboxSettings=lambda: types.SimpleNamespace(
            prompt_repo_layout_lines=0,
            prompt_chunk_token_threshold=20,
            chunk_summary_cache_dir="cache",
            prompt_chunk_cache_dir="cache",
            audit_log_path="audit.log",
            audit_privkey=None,
            prompt_success_log_path="s.log",
            prompt_failure_log_path="f.log",
        ),
        load_sandbox_settings=lambda: types.SimpleNamespace(
            prompt_repo_layout_lines=0,
            prompt_chunk_token_threshold=20,
            chunk_summary_cache_dir="cache",
            prompt_chunk_cache_dir="cache",
            audit_log_path="audit.log",
            audit_privkey=None,
            prompt_success_log_path="s.log",
            prompt_failure_log_path="f.log",
        ),
    ),
)

_setmod(
    "self_improvement",
    types.ModuleType("self_improvement"),
)
_setmod(
    "self_improvement.prompt_memory",
    types.SimpleNamespace(log_prompt_attempt=lambda *a, **k: None),
)
_spec_tr = importlib.util.spec_from_file_location(
    "self_improvement.target_region", ROOT / "self_improvement" / "target_region.py"  # path-ignore
)
tr_module = importlib.util.module_from_spec(_spec_tr)
sys.modules.setdefault("self_improvement.target_region", tr_module)
sys.modules.setdefault("menace_sandbox.self_improvement.target_region", tr_module)
_spec_tr.loader.exec_module(tr_module)  # type: ignore[attr-defined]


class _DummyBaselineTracker:
    def __init__(self, *a, **k) -> None:
        self._vals: dict[str, float] = {}

    def update(self, **k) -> None:
        self._vals.update(k)

    def get(self, key: str) -> float:
        return self._vals.get(key, 0.0)

    def std(self, key: str) -> float:
        return 1.0


_setmod(
    "self_improvement.baseline_tracker",
    types.SimpleNamespace(BaselineTracker=_DummyBaselineTracker, TRACKER={}),
)

_setmod(
    "self_improvement.init",
    types.SimpleNamespace(
        FileLock=type(
            "_FL",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: None,
            },
        ),
        _atomic_write=lambda *a, **k: None,
    ),
)

roi_mod = types.ModuleType("roi_tracker")
roi_mod.ROITracker = lambda: object()
_setmod("roi_tracker", roi_mod)


import menace_sandbox.self_coding_engine as sce  # noqa: E402
from chunking import CodeChunk  # noqa: E402
from chunk_summary_cache import ChunkSummaryCache  # noqa: E402
import chunking as pc  # noqa: E402


def test_generate_helper_injects_chunk_summaries(monkeypatch, tmp_path):
    # Force token count to exceed threshold so chunking is triggered
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    called: dict[str, int] = {}

    def fake_get_chunk_summaries(
        path: Path, limit: int, llm=None, cache=None, context_builder=None
    ):
        called["limit"] = limit
        return [
            {"start_line": 1, "end_line": 2, "summary": "sum1"},
            {"start_line": 3, "end_line": 4, "summary": "sum2"},
        ]

    monkeypatch.setattr(pc, "get_chunk_summaries", fake_get_chunk_summaries)
    monkeypatch.setattr(sce, "get_chunk_summaries", fake_get_chunk_summaries)

    captured: dict[str, object] = {}

    class DummyPrompt:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.system = ""
            self.examples: list[str] = []

    class DummyPromptEngine:
        def build_prompt(
            self,
            goal,
            *,
            context=None,
            retrieval_context=None,
            retry_trace=None,
            tone=None,
            summaries=None,
        ):
            captured["context"] = context
            captured["summaries"] = summaries
            return DummyPrompt()

    monkeypatch.setattr(sce, "PromptEngine", lambda *a, **k: DummyPromptEngine())

    class DummyLLM:
        gpt_memory = None

        def generate(self, prompt):
            return types.SimpleNamespace(text="")

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyLLM(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)

    monkeypatch.setattr(engine, "suggest_snippets", lambda desc, limit=3: [])
    monkeypatch.setattr(engine, "_get_repo_layout", lambda lines: "")

    target = tmp_path / "big.py"  # path-ignore
    target.write_text("print('hi')\n")

    engine.generate_helper("do something", path=target)

    assert called["limit"] == engine.chunk_token_threshold
    assert "Chunk 0: sum1" in captured["context"]
    assert "Chunk 1: sum2" in captured["context"]
    assert captured["summaries"] == ["Chunk 0: sum1", "Chunk 1: sum2"]


def test_generate_helper_uses_cached_chunk_summaries(monkeypatch, tmp_path):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    def fake_chunk_file(path: Path, limit: int):
        return [
            CodeChunk(1, 2, "code1", "h1", 5),
            CodeChunk(3, 4, "code2", "h2", 5),
        ]

    monkeypatch.setattr(pc, "chunk_file", fake_chunk_file)

    calls = {"n": 0}

    def fake_summary(text: str, llm):
        calls["n"] += 1
        return f"sum:{text}"

    monkeypatch.setattr(pc, "summarize_code", fake_summary)
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(tmp_path))

    class DummyPrompt:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.system = ""
            self.examples: list[str] = []

    class DummyPromptEngine:
        def build_prompt(self, *a, **k):
            return DummyPrompt()

    monkeypatch.setattr(sce, "PromptEngine", lambda *a, **k: DummyPromptEngine())

    class DummyLLM:
        gpt_memory = None

        def generate(self, prompt):
            return types.SimpleNamespace(text="")

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyLLM(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )

    monkeypatch.setattr(engine, "suggest_snippets", lambda desc, limit=3: [])
    monkeypatch.setattr(engine, "_get_repo_layout", lambda lines: "")

    target = tmp_path / "big.py"  # path-ignore
    target.write_text("print('hi')\n")

    engine.generate_helper("do something", path=target)
    engine.generate_helper("do something", path=target)

    assert calls["n"] == 2  # summaries computed once and then served from cache


def test_generate_helper_builds_line_range_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 0)

    captured: dict[str, str] = {}

    class DummyPrompt:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.system = ""
            self.examples: list[str] = []

    class DummyPromptEngine:
        def build_prompt(
            self,
            goal,
            *,
            context=None,
            retrieval_context=None,
            retry_trace=None,
            tone=None,
            summaries=None,
        ):
            captured["context"] = context
            return DummyPrompt()

    monkeypatch.setattr(sce, "PromptEngine", lambda *a, **k: DummyPromptEngine())

    class DummyLLM:
        gpt_memory = None

        def generate(self, prompt):
            return types.SimpleNamespace(text="")

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyLLM(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)

    monkeypatch.setattr(engine, "suggest_snippets", lambda desc, limit=3: [])
    monkeypatch.setattr(engine, "_get_repo_layout", lambda lines: "")

    target = tmp_path / "mod.py"  # path-ignore
    target.write_text("a=1\nb=2\nc=3\n")
    region = sce.TargetRegion(start_line=2, end_line=2, function="f")

    engine.generate_helper("do something", path=target, target_region=region)

    ctx = captured["context"]
    assert "Modify only lines 2-2" in ctx
    assert "# start\nb=2\n# end" in ctx
    assert "a=1" not in ctx


def test_patch_file_uses_chunk_summaries(monkeypatch, tmp_path):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    def fake_get_chunk_summaries(
        path: Path, limit: int, llm=None, cache=None, context_builder=None
    ):
        return [
            {"start_line": 1, "end_line": 1, "summary": "sum1"},
            {"start_line": 2, "end_line": 2, "summary": "sum2"},
        ]

    monkeypatch.setattr(pc, "get_chunk_summaries", fake_get_chunk_summaries)
    monkeypatch.setattr(sce, "get_chunk_summaries", fake_get_chunk_summaries)

    captured: dict[str, object] = {}

    class DummyPrompt:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.system = ""
            self.examples: list[str] = []

    class DummyLLM:
        gpt_memory = None

        def generate(self, prompt):
            return types.SimpleNamespace(text="")

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyLLM(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)

    def fake_build_prompt(goal, *, context=None, summaries=None, **kwargs):
        captured["context"] = context
        captured["summaries"] = summaries
        return DummyPrompt()

    monkeypatch.setattr(engine.prompt_engine, "build_prompt", fake_build_prompt)

    target = tmp_path / "big.py"  # path-ignore
    target.write_text("a\nb\n")

    engine.patch_file(target, "desc")

    assert "Chunk 0: sum1" in captured["context"]
    assert "Chunk 1: sum2" in captured["context"]
    assert captured["summaries"] == ["Chunk 0: sum1", "Chunk 1: sum2"]


def test_patch_file_rejects_scope_violation(tmp_path):
    events: list[tuple[str, dict[str, object]]] = []

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)
    engine.event_bus = types.SimpleNamespace(
        publish=lambda name, data: events.append((name, data))
    )
    engine.generate_helper = lambda desc, **_: "print('x')\nprint('y')\nprint('z')\n"

    target = tmp_path / "mod.py"  # path-ignore
    target.write_text("print('a')\nprint('b')\nprint('c')\n")
    region = sce.TargetRegion(start_line=2, end_line=2, function="f")

    with pytest.raises(ValueError):
        engine.patch_file(target, "desc", target_region=region)
    assert events and events[0][0] == "patch:scope_violation"


def test_build_file_context_stitches_target_region(tmp_path, monkeypatch):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 100)

    def fake_summary(text: str, llm=None) -> str:
        return text.splitlines()[0]

    monkeypatch.setattr(sce, "summarize_code", fake_summary)
    monkeypatch.setattr(pc, "summarize_code", fake_summary)

    engine = sce.SelfCodingEngine(
        object(),
        object(),
        prompt_chunk_token_threshold=10,
        chunk_summary_cache_dir=tmp_path,
        context_builder=builder,
    )

    path = tmp_path / "mod.py"  # path-ignore
    path.write_text(
        "\n".join(
            [
                "def a():",
                "    pass",
                "",
                "def b():",
                "    pass",
                "",
                "def c():",
                "    pass",
            ]
        )
    )

    region = sce.TargetRegion(start_line=4, end_line=8, function="bc")
    snippet, summaries = engine._build_file_context(path, target_region=region)
    assert "def b():" in snippet and "def c():" in snippet
    assert summaries == ["Chunk 0: def a():"]
