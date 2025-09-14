import types
import sys
import importlib.util
from pathlib import Path
import logging

import pytest
from dynamic_path_router import resolve_path

# Minimal stubs for heavy dependencies
cd_stub = types.SimpleNamespace(
    CodeDB=object,
    CodeRecord=object,
    PatchHistoryDB=object,
    PatchRecord=object,
)
sys.modules.setdefault("code_database", cd_stub)
sys.modules.setdefault("menace.code_database", cd_stub)

sys.modules.setdefault(
    "gpt_memory", types.SimpleNamespace(GPTMemoryManager=object, INSIGHT="INSIGHT", _summarise_text=lambda text, *a, **k: text)
)
sys.modules.setdefault(
    "gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object)
)
sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, DBRouter=object, init_db_router=lambda *a, **k: None)
)
sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(CognitionLayer=object, SharedVectorService=object, ContextBuilder=object)
)
sys.modules.setdefault("menace.trend_predictor", types.SimpleNamespace(TrendPredictor=object))
sys.modules.setdefault("trend_predictor", sys.modules["menace.trend_predictor"])
sys.modules.setdefault("menace.safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
sys.modules.setdefault("safety_monitor", sys.modules["menace.safety_monitor"])
sys.modules.setdefault(
    "llm_interface",
    types.SimpleNamespace(
        Prompt=object,
        LLMResult=types.SimpleNamespace,
        LLMClient=object,
        Completion=types.SimpleNamespace,
    ),
)
llm_router_stub = types.SimpleNamespace(client_from_settings=lambda *a, **k: None)
sys.modules.setdefault("llm_router", llm_router_stub)
sys.modules.setdefault("menace.llm_router", llm_router_stub)
sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=object)
)
sys.modules.setdefault(
    "rate_limit", types.SimpleNamespace(estimate_tokens=lambda *a, **k: 0)
)
sys.modules.setdefault(
    "shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
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
sys.modules.setdefault("menace.menace_sanity_layer", types.SimpleNamespace(fetch_recent_billing_issues=lambda: ""))
sys.modules.setdefault("menace_sanity_layer", sys.modules["menace.menace_sanity_layer"])
sys.modules.setdefault(
    "patch_suggestion_db", types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object)
)
sys.modules.setdefault(
    "prompt_engine",
    types.SimpleNamespace(
        PromptEngine=lambda *a, **k: types.SimpleNamespace(last_metadata={}),
        _ENCODER=None,
        diff_within_target_region=lambda *a, **k: False,
        build_prompt=lambda *a, **k: object(),
    ),
)
sys.modules.setdefault(
    "chunking",
    types.SimpleNamespace(
        split_into_chunks=lambda *a, **k: [],
        get_chunk_summaries=lambda *a, **k: [],
    ),
)
sys.modules.setdefault(
    "failure_retry_utils", types.SimpleNamespace(check_similarity_and_warn=lambda *a, **k: None, record_failure=lambda *a, **k: None)
)
sys.modules.setdefault(
    "failure_fingerprint", types.SimpleNamespace(FailureFingerprint=object, find_similar=lambda *a, **k: [], log_fingerprint=lambda *a, **k: None)
)
sys.modules.setdefault(
    "metrics_exporter", types.SimpleNamespace(Gauge=lambda *a, **k: types.SimpleNamespace(labels=lambda *a, **k: None))
)
sys.modules.setdefault(
    "codex_fallback_handler", types.SimpleNamespace(handle=lambda *a, **k: types.SimpleNamespace(text=""))
)
sys.modules.setdefault(
    "self_improvement.baseline_tracker", types.SimpleNamespace(BaselineTracker=object, TRACKER=object)
)
sys.modules.setdefault(
    "self_improvement.init", types.SimpleNamespace(FileLock=object, _atomic_write=lambda *a, **k: None)
)

# Prepare package context for relative imports
spec = importlib.util.spec_from_file_location("menace", resolve_path("__init__.py"))
menace_pkg = importlib.util.module_from_spec(spec)
menace_pkg.__path__ = [str(Path().resolve())]
sys.modules.setdefault("menace", menace_pkg)
spec.loader.exec_module(menace_pkg)

spec = importlib.util.spec_from_file_location("menace.self_coding_engine", resolve_path("self_coding_engine.py"))
self_coding_engine = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace.self_coding_engine", self_coding_engine)
spec.loader.exec_module(self_coding_engine)

SelfCodingEngine = self_coding_engine.SelfCodingEngine
Prompt = self_coding_engine.Prompt


class DummyBuilder:
    def build_prompt(self, goal: str, **kwargs):
        return Prompt(user=goal, metadata={"vectors": [("p", 0.5)], "vector_confidences": [0.5]})


def test_enriched_prompt_merges_metadata():
    engine = SelfCodingEngine.__new__(SelfCodingEngine)
    engine.context_builder = DummyBuilder()
    engine._last_retry_trace = "trace"
    engine._last_prompt_metadata = {}
    prompt = engine.build_enriched_prompt(
        "do things",
        context_builder=engine.context_builder,
        intent_metadata={"intent": "meta"},
    )
    assert prompt.metadata["intent"] == "meta"
    assert prompt.metadata["error_log"] == "trace"
    assert "vectors" in prompt.metadata
    assert engine._last_prompt is prompt


def test_llm_requires_enriched_prompt():
    engine = SelfCodingEngine.__new__(SelfCodingEngine)
    engine.llm_client = types.SimpleNamespace()
    engine.logger = logging.getLogger("t")
    engine._last_prompt = None
    with pytest.raises(RuntimeError):
        engine._invoke_llm()
