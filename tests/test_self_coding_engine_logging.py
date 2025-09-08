# flake8: noqa
import logging
import sys
import types
from pathlib import Path
import dynamic_path_router

from llm_interface import LLMResult

# Lightweight stubs to satisfy imports
vec_mod = types.ModuleType("vector_service")
class _VSError(Exception):
    pass
vec_mod.CognitionLayer = object
vec_mod.PatchLogger = object
vec_mod.VectorServiceError = _VSError
vec_mod.EmbeddableDBMixin = object
vec_mod.SharedVectorService = object
vec_mod.ContextBuilder = object
sys.modules.setdefault("vector_service", vec_mod)
retr_mod = types.ModuleType("vector_service.retriever")
retr_mod.Retriever = lambda: None
retr_mod.FallbackResult = list
sys.modules.setdefault("vector_service.retriever", retr_mod)

sys.modules.setdefault("vector_service.decorators", types.ModuleType("vector_service.decorators"))

# Minimal stubs for modules referenced by SelfCodingEngine
code_db_mod = types.ModuleType("code_database")
code_db_mod.CodeDB = object
code_db_mod.CodeRecord = object
code_db_mod.PatchHistoryDB = object
code_db_mod.PatchRecord = object
sys.modules.setdefault("code_database", code_db_mod)
sys.modules.setdefault("menace.code_database", code_db_mod)

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
sys.modules.setdefault("trend_predictor", types.SimpleNamespace(TrendPredictor=object))
sys.modules.setdefault("gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object))
sys.modules.setdefault("safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
sys.modules.setdefault("advanced_error_management", types.SimpleNamespace(FormalVerifier=object))
sys.modules.setdefault("chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault("menace.chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault("memory_aware_gpt_client", types.SimpleNamespace(ask_with_memory=lambda *a, **k: {}))
sys.modules.setdefault("shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None))
log_tags_mod = types.SimpleNamespace(
    FEEDBACK="feedback",
    ERROR_FIX="error_fix",
    IMPROVEMENT_PATH="improvement_path",
    INSIGHT="insight",
)
sys.modules.setdefault("log_tags", log_tags_mod)
class _GKS:
    def __init__(self, *a, **k):
        pass

sys.modules.setdefault("gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=_GKS))
know_mod = types.ModuleType("knowledge_retriever")
know_mod.get_feedback = lambda *a, **k: []
know_mod.get_error_fixes = lambda *a, **k: []
know_mod.recent_feedback = lambda *a, **k: None
know_mod.recent_error_fix = lambda *a, **k: None
know_mod.recent_improvement_path = lambda *a, **k: None
sys.modules.setdefault("knowledge_retriever", know_mod)

sys.modules.setdefault("rollback_manager", types.SimpleNamespace(RollbackManager=object))
audit_mod = types.ModuleType("audit_trail")
audit_mod.AuditTrail = lambda *a, **k: types.SimpleNamespace(record=lambda self, payload: None)
sys.modules.setdefault("audit_trail", audit_mod)
access_mod = types.SimpleNamespace(READ="r", WRITE="w", check_permission=lambda *a, **k: None)
sys.modules.setdefault("access_control", access_mod)
sys.modules.setdefault("patch_suggestion_db", types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object))
sys.modules.setdefault("sandbox_runner.workflow_sandbox_runner", types.SimpleNamespace(WorkflowSandboxRunner=object))
sys.modules.setdefault(
    "sandbox_settings",
    types.SimpleNamespace(
        SandboxSettings=lambda: types.SimpleNamespace(
            va_prompt_template="",
            va_prompt_prefix="",
            va_repo_layout_lines=0,
        )
    ),
)
roi_mod = types.ModuleType("roi_tracker")
roi_mod.ROITracker = lambda: object()
sys.modules.setdefault("roi_tracker", roi_mod)

# Import the SelfCodingEngine after stubs
import importlib
menace_pkg = importlib.import_module("menace")
sys.modules["code_database"] = code_db_mod
sys.modules["menace.code_database"] = code_db_mod
import menace.self_coding_engine as sce
SelfCodingEngine = sce.SelfCodingEngine


class DummyMemory:
    def store(self, *args, **kwargs):
        pass


def test_roi_tracker_logging(caplog):
    class BadPatchLogger:
        @property
        def roi_tracker(self):
            return None

        @roi_tracker.setter
        def roi_tracker(self, value):
            raise RuntimeError("fail patch_logger")

    class BadCognitionLayer:
        roi_tracker = None

        def __init__(self, builder):
            self.context_builder = builder

        def __setattr__(self, name, value):
            if name == "roi_tracker":
                raise RuntimeError("fail cognition_layer")
            super().__setattr__(name, value)

    builder = types.SimpleNamespace(
        build_context=lambda *a, **k: {},
        refresh_db_weights=lambda *a, **k: None,
    )
    caplog.set_level(logging.WARNING)
    SelfCodingEngine(
        object(),
        DummyMemory(),
        patch_logger=BadPatchLogger(),
        cognition_layer=BadCognitionLayer(builder),
        context_builder=builder,
    )
    messages = [record.message for record in caplog.records]
    assert any("patch_logger" in m for m in messages)
    assert any("cognition_layer" in m for m in messages)


def test_knowledge_service_logging(monkeypatch, caplog):
    llm = types.SimpleNamespace(gpt_memory=None, generate=lambda prompt: LLMResult(text=""))
    engine = SelfCodingEngine(
        object(),
        DummyMemory(),
        knowledge_service=object(),
        llm_client=llm,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(sce, "get_feedback", lambda *a, **k: [])
    monkeypatch.setattr(sce, "get_error_fixes", lambda *a, **k: [])

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(sce, "recent_feedback", boom)
    monkeypatch.setattr(sce, "recent_improvement_path", boom)
    monkeypatch.setattr(sce, "recent_error_fix", boom)
    monkeypatch.setattr(sce.SelfCodingEngine, "_get_repo_layout", lambda self, lines: "")
    monkeypatch.setattr(sce.PromptEngine, "build_prompt", staticmethod(lambda *a, **k: ""))
    monkeypatch.setattr(
        sce.SelfCodingEngine, "suggest_snippets", lambda self, desc, limit=3: []
    )
    caplog.set_level(logging.WARNING)
    caplog.clear()
    target = dynamic_path_router.resolve_path("tests/fixtures/semantic/a.py")  # path-ignore
    engine.generate_helper("desc", path=target)
    messages = [record.message for record in caplog.records]
    assert any("recent_feedback" in m for m in messages)
    assert any("recent_improvement_path" in m for m in messages)
    assert any("recent_error_fix" in m for m in messages)


def test_tempfile_cleanup_logging(monkeypatch, caplog, tmp_path):
    llm = types.SimpleNamespace(gpt_memory=None)
    engine = SelfCodingEngine(
        object(),
        DummyMemory(),
        llm_client=llm,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )

    class Verifier:
        def verify(self, path):
            return True

    engine.formal_verifier = Verifier()
    monkeypatch.setattr(
        engine,
        "generate_helper",
        lambda description, path=None, metadata=None: "x = 1\n",
    )

    def bad_unlink(self):
        raise RuntimeError("unlink boom")

    monkeypatch.setattr(sce.Path, "unlink", bad_unlink)
    caplog.set_level(logging.ERROR)
    caplog.clear()
    target = tmp_path / "t.py"  # path-ignore
    target.write_text("print('hi')\n")
    engine.patch_file(target, "desc")
    assert any("temporary file cleanup failed" in r.message for r in caplog.records)
