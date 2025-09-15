import logging
import sys
import types
import importlib.util
from pathlib import Path

# Stubs for heavy dependencies before importing the target module
cd_stub = types.SimpleNamespace(
    CodeDB=object,
    CodeRecord=object,
    PatchHistoryDB=object,
    PatchRecord=object,
)
sys.modules.setdefault("code_database", cd_stub)
sys.modules.setdefault("menace.code_database", cd_stub)

sys.modules.setdefault(
    "gpt_memory",
    types.SimpleNamespace(
        GPTMemoryManager=object,
        INSIGHT="INSIGHT",
        _summarise_text=lambda text, *a, **k: text,
    ),
)
sys.modules.setdefault(
    "gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object)
)
sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, DBRouter=object, init_db_router=lambda *a, **k: None)
)
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(CognitionLayer=object, SharedVectorService=object, ContextBuilder=object),
)
sys.modules.setdefault(
    "menace.trend_predictor", types.SimpleNamespace(TrendPredictor=object)
)
sys.modules.setdefault("trend_predictor", sys.modules["menace.trend_predictor"])
sys.modules.setdefault(
    "menace.safety_monitor", types.SimpleNamespace(SafetyMonitor=object)
)
sys.modules.setdefault("safety_monitor", sys.modules["menace.safety_monitor"])
sys.modules.setdefault(
    "llm_interface", types.SimpleNamespace(Prompt=str, LLMResult=object, LLMClient=object)
)
sys.modules.setdefault(
    "llm_registry",
    types.SimpleNamespace(create_backend=lambda *a, **k: None, register_backend_from_path=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=object)
)
sys.modules.setdefault(
    "rate_limit", types.SimpleNamespace(estimate_tokens=lambda *a, **k: 0)
)
sys.modules.setdefault(
    "menace.rate_limit", types.SimpleNamespace(estimate_tokens=lambda *a, **k: 0)
)
sys.modules.setdefault(
    "menace.llm_router", types.SimpleNamespace(client_from_settings=lambda *a, **k: None)
)
sys.modules.setdefault("llm_router", sys.modules["menace.llm_router"])
sys.modules.setdefault(
    "shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "shared_knowledge_module", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
sys.modules.setdefault(
    "gpt_knowledge_service", sys.modules["menace.gpt_knowledge_service"]
)
sys.modules.setdefault(
    "governed_retrieval", types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x)
)
sys.modules.setdefault(
    "menace.roi_tracker", types.SimpleNamespace(ROITracker=type("RT", (), {}))
)
sys.modules.setdefault("roi_tracker", sys.modules["menace.roi_tracker"])
sys.modules.setdefault(
    "knowledge_retriever",
    types.SimpleNamespace(
        get_feedback=lambda *a, **k: [],
        get_error_fixes=lambda *a, **k: [],
        recent_feedback=lambda *a, **k: "",
        recent_error_fix=lambda *a, **k: "",
        recent_improvement_path=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.menace_sanity_layer", types.SimpleNamespace(fetch_recent_billing_issues=lambda: "")
)
sys.modules.setdefault("menace_sanity_layer", sys.modules["menace.menace_sanity_layer"])
sys.modules.setdefault(
    "menace.patch_suggestion_db", types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object)
)
sys.modules.setdefault("patch_suggestion_db", sys.modules["menace.patch_suggestion_db"])
sys.modules.setdefault(
    "self_improvement.init", types.SimpleNamespace(FileLock=object, _atomic_write=lambda *a, **k: None)
)
sys.modules.setdefault(
    "sandbox_runner.bootstrap", types.SimpleNamespace(initialize_autonomous_sandbox=lambda *a, **k: None)
)

spec = importlib.util.spec_from_file_location(
    "menace.self_coding_engine", Path(__file__).resolve().parents[1] / "self_coding_engine.py"
)
sce = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace.self_coding_engine", sce)
spec.loader.exec_module(sce)


def test_log_prompt_attempt_logging(monkeypatch, caplog):
    class DummyBuilder:
        def build_prompt(self, query, **_):
            return types.SimpleNamespace(user="bob", metadata={})

        def refresh_db_weights(self):
            return None

    class DummyClient:
        def generate(self, prompt):
            return types.SimpleNamespace(text="ok")

    builder = DummyBuilder()
    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyClient(),
        gpt_memory=types.SimpleNamespace(search_context=lambda *a, **k: []),
        context_builder=builder,
        roi_tracker=object(),
    )

    def boom(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr(sce, "log_prompt_attempt", boom)
    with caplog.at_level(logging.ERROR):
        engine.build_enriched_prompt("task", context_builder=builder)
    assert any("log_prompt_attempt failed" in r.message for r in caplog.records)
