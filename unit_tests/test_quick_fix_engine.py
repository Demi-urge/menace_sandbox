import sys
import types
import contextvars
import pytest


@pytest.fixture
def qfe(monkeypatch):
    sr_mod = types.ModuleType("sandbox_runner")
    sr_mod.post_round_orphan_scan = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)

    eb_mod = types.ModuleType("menace_sandbox.error_bot")
    class ErrorDB: ...
    eb_mod.ErrorDB = ErrorDB
    monkeypatch.setitem(sys.modules, "menace_sandbox.error_bot", eb_mod)

    scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")
    class SelfCodingManager:
        def validate_provenance(self, token):
            pass
    scm_mod.SelfCodingManager = SelfCodingManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", scm_mod)

    sce_mod = types.ModuleType("menace_sandbox.self_coding_engine")
    class SelfCodingEngine: ...
    sce_mod.SelfCodingEngine = SelfCodingEngine
    sce_mod.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=None)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", sce_mod)

    kg_mod = types.ModuleType("menace_sandbox.knowledge_graph")
    class KnowledgeGraph: ...
    kg_mod.KnowledgeGraph = KnowledgeGraph
    monkeypatch.setitem(sys.modules, "menace_sandbox.knowledge_graph", kg_mod)

    cbi_mod = types.ModuleType("menace_sandbox.coding_bot_interface")
    def manager_generate_helper(manager, description: str, **kwargs):
        return manager.engine.generate_helper(description, **kwargs)
    cbi_mod.manager_generate_helper = manager_generate_helper
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi_mod)

    vec_cb_mod = types.ModuleType("vector_service.context_builder")
    class ContextBuilder: ...
    class Retriever: ...
    class FallbackResult(str): ...
    class EmbeddingBackfill:
        def run(self, *a, **k):
            pass
    vec_cb_mod.ContextBuilder = ContextBuilder
    vec_cb_mod.Retriever = Retriever
    vec_cb_mod.FallbackResult = FallbackResult
    vec_cb_mod.EmbeddingBackfill = EmbeddingBackfill
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", vec_cb_mod)

    vec_mod = types.ModuleType("vector_service")
    class ErrorResult(Exception): ...
    vec_mod.ErrorResult = ErrorResult
    monkeypatch.setitem(sys.modules, "vector_service", vec_mod)

    patch_mod = types.ModuleType("patch_provenance")
    class PatchLogger:
        def __init__(self, *a, **k):
            pass

        def track_contributors(self, *a, **k):
            pass
    patch_mod.PatchLogger = PatchLogger
    monkeypatch.setitem(sys.modules, "patch_provenance", patch_mod)

    cb_util_mod = types.ModuleType("context_builder_util")
    def ensure_fresh_weights(builder):
        builder.refresh_db_weights()
    cb_util_mod.ensure_fresh_weights = ensure_fresh_weights
    monkeypatch.setitem(sys.modules, "context_builder_util", cb_util_mod)

    data_bot_mod = types.ModuleType("menace_sandbox.data_bot")
    class DataBot: ...
    data_bot_mod.DataBot = DataBot
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot_mod)

    resilience_mod = types.ModuleType("menace_sandbox.resilience")
    resilience_mod.retry_with_backoff = lambda fn, *a, **k: fn()
    monkeypatch.setitem(sys.modules, "menace_sandbox.resilience", resilience_mod)

    aem_mod = types.ModuleType("menace_sandbox.advanced_error_management")
    class AutomatedRollbackManager: ...
    aem_mod.AutomatedRollbackManager = AutomatedRollbackManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.advanced_error_management", aem_mod)

    code_db_mod = types.ModuleType("menace_sandbox.code_database")
    class PatchHistoryDB: ...
    code_db_mod.PatchHistoryDB = PatchHistoryDB
    monkeypatch.setitem(sys.modules, "menace_sandbox.code_database", code_db_mod)

    trg_mod = types.ModuleType("menace_sandbox.target_region")
    trg_mod.extract_target_region = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.target_region", trg_mod)

    chunk_mod = types.ModuleType("chunking")
    chunk_mod.get_chunk_summaries = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "chunking", chunk_mod)

    ps_mod = types.ModuleType("menace_sandbox.self_improvement.prompt_strategies")
    class PromptStrategy(str): ...
    def render_prompt(*a, **k):
        return ""
    ps_mod.PromptStrategy = PromptStrategy
    ps_mod.render_prompt = render_prompt
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_improvement.prompt_strategies", ps_mod
    )

    cdc_mod = types.ModuleType("menace_sandbox.codebase_diff_checker")
    cdc_mod.generate_code_diff = lambda *a, **k: {}
    cdc_mod.flag_risky_changes = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "menace_sandbox.codebase_diff_checker", cdc_mod)

    haf_mod = types.ModuleType("menace_sandbox.human_alignment_flagger")
    haf_mod._collect_diff_data = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "menace_sandbox.human_alignment_flagger", haf_mod)

    haa_mod = types.ModuleType("menace_sandbox.human_alignment_agent")
    haa_mod.HumanAlignmentAgent = object
    monkeypatch.setitem(sys.modules, "menace_sandbox.human_alignment_agent", haa_mod)

    vl_mod = types.ModuleType("menace_sandbox.violation_logger")
    vl_mod.log_violation = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.violation_logger", vl_mod)

    import menace_sandbox.quick_fix_engine as qfe_mod

    monkeypatch.setattr(qfe_mod, "get_chunk_summaries", None)
    return qfe_mod


def test_context_block_compressed(qfe):
    class DummyPatchLogger:
        def track_contributors(self, *a, **k):
            raise RuntimeError("skip")

    context_block = "a" * 250

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            return context_block, "", []

    class DummyEngine:
        def __init__(self):
            self.helper_calls: list[str] = []
            self.patched: dict[str, str] = {}

        def generate_helper(self, desc, **kwargs):
            self.helper_calls.append(desc)
            return "helper"

        def apply_patch_with_retry(self, path, helper, **kwargs):
            self.patched["helper"] = helper
            self.patched["desc"] = kwargs.get("description", "")
            return 1, "", ""

    builder = DummyBuilder()
    engine = DummyEngine()
    manager = qfe.SelfCodingManager()
    manager.engine = engine
    manager.register_patch_cycle = lambda *a, **k: None

    expected = qfe.compress_snippets({"snippet": context_block})["snippet"]

    qfe.generate_patch(
        module="simple_functions",
        manager=manager,
        engine=engine,
        context_builder=builder,
        provenance_token="tok",
        description="desc",
        patch_logger=DummyPatchLogger(),
    )

    assert engine.helper_calls == [f"desc\n\n{expected}"]
    assert engine.patched["helper"] == "helper"
    assert engine.patched["desc"] == f"desc\n\n{expected}"
    assert context_block not in engine.helper_calls[0]


def test_generate_patch_errors_without_manager(qfe):
    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, *a, **k):
            return ""

    with pytest.raises(RuntimeError):
        qfe.generate_patch(
            "mod", None, context_builder=DummyBuilder(), provenance_token="tok"
        )


def test_generate_patch_errors_with_wrong_manager(qfe):
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, *a, **k):
            return ""

    with pytest.raises(RuntimeError):
        qfe.generate_patch(
            "mod", object(), context_builder=DummyBuilder(), provenance_token="tok"
        )


def test_generate_patch_requires_explicit_manager(qfe):
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, *a, **k):
            return ""

    qfe.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=object())
    with pytest.raises(RuntimeError):
        qfe.generate_patch(
            "mod", None, context_builder=DummyBuilder(), provenance_token="tok"
        )


def test_generate_patch_rejects_unmanaged(qfe):
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, *a, **k):
            return ""

    class DummyEngine:
        def apply_patch_with_retry(self, path, helper, **kwargs):
            return 1, "", ""

    engine = DummyEngine()
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    with pytest.raises(RuntimeError):
        qfe.generate_patch(
            "mod",
            manager,
            engine=engine,
            context_builder=DummyBuilder(),
            provenance_token="tok",
        )


def test_generate_patch_enriches_with_graph(qfe):
    captured: dict[str, str] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            return "", "", []

    class DummyEngine:
        def apply_patch_with_retry(self, path, helper, **kwargs):
            return 1, "", ""

    def helper_fn(manager, desc, **kwargs):
        captured["desc"] = desc
        return "helper"

    class DummyGraph:
        def related(self, key: str, depth: int = 1) -> list[str]:
            return ["code:foo.py", "error:ValueError"]

    builder = DummyBuilder()
    engine = DummyEngine()
    manager = qfe.SelfCodingManager()
    manager.engine = engine
    manager.register_patch_cycle = lambda *a, **k: None

    qfe.generate_patch(
        module="simple_functions",
        manager=manager,
        engine=engine,
        context_builder=builder,
        provenance_token="tok",
        description="desc",
        helper_fn=helper_fn,
        graph=DummyGraph(),
    )

    assert "foo.py" in captured["desc"]
    assert "ValueError" in captured["desc"]


def test_generate_patch_graph_failure(qfe):
    captured: dict[str, str] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            return "", "", []

    class DummyEngine:
        def apply_patch_with_retry(self, path, helper, **kwargs):
            return 1, "", ""

    def helper_fn(manager, desc, **kwargs):
        captured["desc"] = desc
        return "helper"

    class FailingGraph:
        def related(self, *a, **k):
            raise RuntimeError("boom")

    builder = DummyBuilder()
    engine = DummyEngine()
    manager = qfe.SelfCodingManager()
    manager.engine = engine
    manager.register_patch_cycle = lambda *a, **k: None

    pid = qfe.generate_patch(
        module="simple_functions",
        manager=manager,
        engine=engine,
        context_builder=builder,
        provenance_token="tok",
        description="desc",
        helper_fn=helper_fn,
        graph=FailingGraph(),
    )

    assert pid == 1
    assert "Related modules" not in captured["desc"]
