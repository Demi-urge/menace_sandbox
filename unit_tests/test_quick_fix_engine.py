import sys
import types
import pytest


def test_context_block_compressed(monkeypatch):
    sr_mod = types.ModuleType("sandbox_runner")
    sr_mod.post_round_orphan_scan = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)

    eb_mod = types.ModuleType("menace_sandbox.error_bot")
    class ErrorDB: ...
    eb_mod.ErrorDB = ErrorDB
    monkeypatch.setitem(sys.modules, "menace_sandbox.error_bot", eb_mod)

    scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")
    class SelfCodingManager: ...
    scm_mod.SelfCodingManager = SelfCodingManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", scm_mod)

    kg_mod = types.ModuleType("menace_sandbox.knowledge_graph")
    class KnowledgeGraph: ...
    kg_mod.KnowledgeGraph = KnowledgeGraph
    monkeypatch.setitem(sys.modules, "menace_sandbox.knowledge_graph", kg_mod)

    ecp_mod = types.ModuleType("menace_sandbox.error_cluster_predictor")
    ecp_mod.ErrorClusterPredictor = object
    monkeypatch.setitem(sys.modules, "menace_sandbox.error_cluster_predictor", ecp_mod)

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

    import menace_sandbox.quick_fix_engine as qfe

    monkeypatch.setattr(qfe, "get_chunk_summaries", None)

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
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    expected = qfe.compress_snippets({"snippet": context_block})["snippet"]

    qfe.generate_patch(
        module="simple_functions",
        manager=manager,
        engine=engine,
        context_builder=builder,
        description="desc",
        patch_logger=DummyPatchLogger(),
    )

    assert engine.helper_calls == [f"desc\n\n{expected}"]
    assert engine.patched["helper"] == "helper"
    assert engine.patched["desc"] == f"desc\n\n{expected}"
    assert context_block not in engine.helper_calls[0]


def test_generate_patch_errors_without_manager(monkeypatch):
    import menace_sandbox.quick_fix_engine as qfe

    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, *a, **k):
            return ""

    with pytest.raises(RuntimeError):
        qfe.generate_patch("mod", None, context_builder=DummyBuilder())

