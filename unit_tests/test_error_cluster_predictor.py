import sys
import types
import sqlite3


def _setup_modules(monkeypatch):
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
    class PatchLogger: ...
    patch_mod.PatchLogger = PatchLogger
    monkeypatch.setitem(sys.modules, "patch_provenance", patch_mod)

    chunk_mod = types.ModuleType("chunking")
    chunk_mod.get_chunk_summaries = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "chunking", chunk_mod)

    ps_mod = types.ModuleType("menace_sandbox.self_improvement.prompt_strategies")
    class PromptStrategy(str): ...
    ps_mod.PromptStrategy = PromptStrategy
    ps_mod.render_prompt = lambda *a, **k: ""
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


def _make_db(traces):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE telemetry (module text, stack_trace text)")
    conn.executemany(
        "INSERT INTO telemetry (module, stack_trace) VALUES (?, ?)",
        [("mod", t) for t in traces],
    )
    return types.SimpleNamespace(conn=conn)


def test_best_cluster_groups_similar_traces(monkeypatch):
    _setup_modules(monkeypatch)
    from menace_sandbox.quick_fix_engine import ErrorClusterPredictor

    traces = [
        "ValueError: x\nline1\nline2",
        "ValueError: x\nline1\nline2",
        "TypeError: y\nother line",
    ]
    db = _make_db(traces)
    predictor = ErrorClusterPredictor(db)
    cluster_id, cluster_traces, size = predictor.best_cluster("mod", n_clusters=2)
    assert cluster_id in (0, 1)
    assert size == 2
    assert len(cluster_traces) == size
    assert all("ValueError" in t for t in cluster_traces)
