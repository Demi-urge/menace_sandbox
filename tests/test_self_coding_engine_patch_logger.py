import importlib.util
import sys
import types
from pathlib import Path
import dynamic_path_router

import pytest


class DummyROITracker:
    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, float]] = {}

    def update_db_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        self.metrics.update(metrics)

    def origin_db_deltas(self) -> dict[str, float]:
        return {k: v.get("roi", 0.0) for k, v in self.metrics.items()}


class RecordingPatchLogger:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.roi_tracker: DummyROITracker | None = None

    def track_contributors(self, vector_ids, result, **kwargs):
        self.calls.append((vector_ids, result, kwargs))
        tracker = self.roi_tracker
        if tracker is None:
            return {}
        contrib = kwargs.get("roi_delta", kwargs.get("contribution", 0.0))
        totals: dict[str, float] = {}
        for vid in vector_ids:
            if isinstance(vid, tuple):
                vid = vid[0]
            origin = vid.split(":", 1)[0] if ":" in vid else ""
            totals[origin] = totals.get(origin, 0.0) + contrib
        tracker.update_db_metrics({o: {"roi": r} for o, r in totals.items()})
        return {}


def load_engine_module():
    def add_stub(name: str, obj: types.ModuleType | object) -> None:
        sys.modules[name] = obj
        sys.modules[f"menace.{name}"] = obj

    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules.setdefault("menace", pkg)
    vec_mod = types.ModuleType("vector_service")
    vec_mod.CognitionLayer = object
    vec_mod.PatchLogger = object
    vec_mod.VectorServiceError = Exception
    vec_mod.ContextBuilder = object
    add_stub("vector_service", vec_mod)
    add_stub(
        "vector_service.decorators",
        types.SimpleNamespace(log_and_measure=lambda f: f, _CALL_COUNT=None, _LATENCY_GAUGE=None, _RESULT_SIZE_GAUGE=None),
    )
    code_db_stub = types.ModuleType("code_database")
    code_db_stub.CodeDB = object
    code_db_stub.CodeRecord = object
    code_db_stub.PatchHistoryDB = object
    code_db_stub.PatchRecord = object
    add_stub("code_database", code_db_stub)
    add_stub("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    add_stub("trend_predictor", types.SimpleNamespace(TrendPredictor=object))
    add_stub("gpt_memory_interface", types.SimpleNamespace(GPTMemoryInterface=object))
    add_stub("safety_monitor", types.SimpleNamespace(SafetyMonitor=object))
    add_stub("advanced_error_management", types.SimpleNamespace(FormalVerifier=object))
    add_stub("chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
    add_stub("memory_aware_gpt_client", types.SimpleNamespace(ask_with_memory=lambda *a, **k: None))
    add_stub("shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None))
    add_stub("menace_sanity_layer", types.SimpleNamespace(fetch_recent_billing_issues=lambda: []))
    add_stub(
        "log_tags",
        types.SimpleNamespace(FEEDBACK="feedback", ERROR_FIX="error_fix", IMPROVEMENT_PATH="imp", INSIGHT="insight"),
    )
    add_stub("gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object))
    def dummy(*a, **k):
        return []
    add_stub(
        "knowledge_retriever",
        types.SimpleNamespace(
            get_feedback=dummy,
            get_error_fixes=dummy,
            recent_feedback=dummy,
            recent_error_fix=dummy,
            recent_improvement_path=dummy,
        ),
    )
    add_stub("rollback_manager", types.SimpleNamespace(RollbackManager=object))
    class DummyAuditTrail:
        def __init__(self, *a, **k):
            pass

    add_stub("audit_trail", types.SimpleNamespace(AuditTrail=DummyAuditTrail))
    add_stub("patch_suggestion_db", types.SimpleNamespace(PatchSuggestionDB=object, SuggestionRecord=object))
    add_stub("sandbox_runner.workflow_sandbox_runner", types.SimpleNamespace(WorkflowSandboxRunner=object))
    add_stub(
        "sandbox_settings",
        types.SimpleNamespace(
            SandboxSettings=lambda: types.SimpleNamespace(
                prompt_repo_layout_lines=0,
                openai_api_key=None,
                audit_log_path="",
                audit_privkey=None,
            )
        ),
    )
    add_stub("roi_tracker", types.SimpleNamespace(ROITracker=DummyROITracker))
    class DummyPromptEngine:
        def __init__(self, *a, **k):
            pass

    add_stub("prompt_engine", types.SimpleNamespace(PromptEngine=DummyPromptEngine))
    path = dynamic_path_router.resolve_path("self_coding_engine.py")  # path-ignore
    spec = importlib.util.spec_from_file_location("menace.self_coding_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["menace.self_coding_engine"] = module
    spec.loader.exec_module(module)
    return module


def test_track_contributors_records_roi():
    sce = load_engine_module()
    class DummyDB: pass
    class DummyMem: pass
    pl = RecordingPatchLogger()
    engine = sce.SelfCodingEngine(
        DummyDB(),
        DummyMem(),
        patch_logger=pl,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    vectors = [("db1", "v1", 0.1), ("db2", "v2", 0.2)]
    engine._track_contributors(
        "sess",
        vectors,
        True,
        roi_delta=1.5,
        retrieval_metadata={"db1:v1": {"prompt_tokens": 3}},
    )
    assert pl.calls
    vids, result, kwargs = pl.calls[0]
    assert kwargs["contribution"] == pytest.approx(1.5)
    assert kwargs["roi_delta"] == pytest.approx(1.5)
    assert kwargs["session_id"] == "sess"
    assert kwargs["retrieval_metadata"]["db1:v1"]["prompt_tokens"] == 3
    assert pl.roi_tracker.metrics["db1"]["roi"] == pytest.approx(1.5)
    assert pl.roi_tracker.metrics["db2"]["roi"] == pytest.approx(1.5)
