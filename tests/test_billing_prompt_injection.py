import sys
import types
import logging
import pytest
from menace.coding_bot_interface import manager_generate_helper

# Stub heavy dependencies before importing the module under test
sys.modules["vector_service"] = types.SimpleNamespace(
    CognitionLayer=object,
    PatchLogger=object,
    VectorServiceError=Exception,
    SharedVectorService=object,
)
sys.modules["menace.code_database"] = types.SimpleNamespace(
    CodeDB=object, CodeRecord=object, PatchHistoryDB=object, PatchRecord=object
)
sys.modules["code_database"] = sys.modules["menace.code_database"]
sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(
    UnifiedEventBus=object
)
sys.modules["unified_event_bus"] = sys.modules["menace.unified_event_bus"]
sys.modules["menace.trend_predictor"] = types.SimpleNamespace(TrendPredictor=object)
sys.modules["trend_predictor"] = sys.modules["menace.trend_predictor"]
sys.modules["menace.shared_gpt_memory"] = types.SimpleNamespace(
    GPT_MEMORY_MANAGER=None
)
sys.modules["shared_gpt_memory"] = sys.modules["menace.shared_gpt_memory"]

import menace.self_coding_engine as sce
from menace.llm_interface import LLMResult


class DummyPrompt:
    def __init__(self, text: str = "base") -> None:
        self.text = text
        self.system = ""
        self.examples = []
        self.metadata = {}

    def __str__(self) -> str:  # pragma: no cover - simple repr
        return self.text


def test_billing_instructions_in_prompt(monkeypatch):
    engine = object.__new__(sce.SelfCodingEngine)
    engine.prompt_engine = types.SimpleNamespace(
        build_prompt=lambda *a, **k: DummyPrompt("start")
    )
    engine.llm_client = object()
    engine.logger = logging.getLogger("test")
    engine.gpt_memory = None
    engine.knowledge_service = None
    engine.formal_verifier = None
    engine.prompt_optimizer = None
    engine.prompt_evolution_memory = None
    engine.roi_tracker = None
    engine.context_builder = types.SimpleNamespace(
        build_context=lambda *a, **k: {},
        refresh_db_weights=lambda *a, **k: None,
    )
    engine.prompt_tone = None
    engine._last_prompt_metadata = {}
    engine._last_retry_trace = ""
    engine.suggest_snippets = lambda *a, **k: []
    engine._build_file_context = lambda *a, **k: ("", None)
    engine._extract_statements = lambda *a, **k: []
    engine._apply_prompt_style = lambda *a, **k: None
    engine._fetch_retry_trace = lambda meta: ""
    engine.simplify_prompt = sce.simplify_prompt

    monkeypatch.setattr(sce, "call_codex_with_backoff", lambda *a, **k: LLMResult(text="pass"))
    monkeypatch.setattr(sce, "get_feedback", lambda *a, **k: [])
    monkeypatch.setattr(sce, "get_error_fixes", lambda *a, **k: [])
    monkeypatch.setattr(sce, "recent_feedback", lambda *a, **k: "")
    monkeypatch.setattr(sce, "recent_improvement_path", lambda *a, **k: "")
    monkeypatch.setattr(sce, "recent_error_fix", lambda *a, **k: "")
    monkeypatch.setattr(
        sce, "fetch_recent_billing_issues", lambda limit=5: ["invoice overdue"]
    )

    manager_generate_helper(types.SimpleNamespace(engine=engine), "demo task")
    assert "invoice overdue" in engine._last_prompt.text
