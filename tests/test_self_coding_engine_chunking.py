from __future__ import annotations

from pathlib import Path
import sys
import types


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
_setmod("vector_service", vec_mod)
_setmod("vector_service.retriever", types.ModuleType("vector_service.retriever"))
_setmod("vector_service.decorators", types.ModuleType("vector_service.decorators"))

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
            va_prompt_template="",
            va_prompt_prefix="",
            va_repo_layout_lines=0,
            prompt_chunk_token_threshold=20,
            chunk_summary_cache_dir="cache",
            prompt_chunk_cache_dir="cache",
            audit_log_path="audit.log",
            audit_privkey=None,
            prompt_success_log_path="s.log",
            prompt_failure_log_path="f.log",
        )
    ),
)

roi_mod = types.ModuleType("roi_tracker")
roi_mod.ROITracker = lambda: object()
_setmod("roi_tracker", roi_mod)


import menace_sandbox.self_coding_engine as sce  # noqa: E402
from chunking import CodeChunk  # noqa: E402


def test_generate_helper_injects_chunk_summaries(monkeypatch, tmp_path):
    # Force token count to exceed threshold so chunking is triggered
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    called: dict[str, int] = {}

    def fake_chunk_file(code: str, limit: int):
        called["limit"] = limit
        return [
            CodeChunk(start_line=1, end_line=2, text="code1", hash="h1", token_count=5),
            CodeChunk(start_line=3, end_line=4, text="code2", hash="h2", token_count=5),
        ]

    monkeypatch.setattr(sce, "split_into_chunks", fake_chunk_file)
    monkeypatch.setattr(sce, "summarize_code", lambda text, llm: f"sum:{text}")

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
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)
    engine.formal_verifier = None

    monkeypatch.setattr(engine, "suggest_snippets", lambda desc, limit=3: [])
    monkeypatch.setattr(engine, "_get_repo_layout", lambda lines: "")
    monkeypatch.setattr(engine, "_build_file_context", lambda path: "raw context")

    target = tmp_path / "big.py"
    target.write_text("print('hi')\n")

    engine.generate_helper("do something", path=target)

    assert called["limit"] == engine.chunk_token_threshold
    assert "sum:code1" in captured["context"]
    assert "sum:code2" in captured["context"]
    assert "raw context" not in captured["context"]

    # Cache file created
    assert list(engine.chunk_cache.cache_dir.glob("*.json"))


def test_generate_helper_resummarizes_cached_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    def fake_chunk_file(code: str, limit: int):
        return [
            CodeChunk(start_line=1, end_line=2, text="code1", hash="h1", token_count=5),
            CodeChunk(start_line=3, end_line=4, text="code2", hash="h2", token_count=5),
        ]

    monkeypatch.setattr(sce, "split_into_chunks", fake_chunk_file)

    calls = {"n": 0}

    def fake_summary(text: str, llm):
        calls["n"] += 1
        return f"sum:{text}".strip()

    monkeypatch.setattr(sce, "summarize_code", fake_summary)

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
    )

    monkeypatch.setattr(engine, "suggest_snippets", lambda desc, limit=3: [])
    monkeypatch.setattr(engine, "_get_repo_layout", lambda lines: "")
    monkeypatch.setattr(engine, "_build_file_context", lambda path: "raw context")

    target = tmp_path / "big.py"
    target.write_text("print('hi')\n")

    engine.generate_helper("do something", path=target)
    engine.generate_helper("do something", path=target)

    assert calls["n"] == 4  # two chunks per call, called twice


def test_patch_file_uses_chunk_summaries(monkeypatch, tmp_path):
    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)

    def fake_split(code: str, limit: int):
        return [
            CodeChunk(1, 2, "code1", "h1", 5),
            CodeChunk(3, 4, "code2", "h2", 5),
        ]

    monkeypatch.setattr(sce, "split_into_chunks", fake_split)
    monkeypatch.setattr(sce, "summarize_code", lambda text, llm: f"sum:{text}")

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
    )
    engine.formal_verifier = None
    engine.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)

    def fake_build_prompt(goal, *, context=None, **kwargs):
        captured["context"] = context
        return DummyPrompt()

    monkeypatch.setattr(engine.prompt_engine, "build_prompt", fake_build_prompt)

    target = tmp_path / "big.py"
    target.write_text("print('hi')\n")

    engine.patch_file(target, "desc")

    assert "sum:code1" in captured["context"]
    assert "sum:code2" in captured["context"]
    assert "print('hi')" not in captured["context"]
