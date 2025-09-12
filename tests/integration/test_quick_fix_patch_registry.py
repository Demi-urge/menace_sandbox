import sys
import types
import importlib.util
from pathlib import Path


def test_quick_fix_registers_registry_and_metrics(tmp_path, monkeypatch):
    ROOT = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT))

    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(ROOT)]
    sys.modules["menace_sandbox"] = package

    dpr = types.ModuleType("dynamic_path_router")
    dpr.resolve_path = lambda p: Path(p)
    dpr.path_for_prompt = lambda p: str(p)
    sys.modules["dynamic_path_router"] = dpr

    sr = types.ModuleType("sandbox_runner")
    sr.post_round_orphan_scan = lambda *a, **k: None
    sys.modules["sandbox_runner"] = sr

    eb = types.ModuleType("menace_sandbox.error_bot")
    eb.ErrorDB = object
    sys.modules["menace_sandbox.error_bot"] = eb

    kg = types.ModuleType("menace_sandbox.knowledge_graph")
    kg.KnowledgeGraph = object
    sys.modules["menace_sandbox.knowledge_graph"] = kg

    scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")

    class SelfCodingManager:  # pragma: no cover - stub
        pass

    scm_mod.SelfCodingManager = SelfCodingManager
    sys.modules["menace_sandbox.self_coding_manager"] = scm_mod

    sce_mod = types.ModuleType("menace_sandbox.self_coding_engine")
    sce_mod.MANAGER_CONTEXT = types.SimpleNamespace()
    sys.modules["menace_sandbox.self_coding_engine"] = sce_mod

    ecp = types.ModuleType("menace_sandbox.error_cluster_predictor")
    ecp.ErrorClusterPredictor = object
    sys.modules["menace_sandbox.error_cluster_predictor"] = ecp

    pp = types.ModuleType("patch_provenance")

    class PatchLogger:
        def __init__(self, *a, **k):
            pass

        def track_contributors(self, *a, **k):
            pass

    pp.PatchLogger = PatchLogger
    sys.modules["patch_provenance"] = pp

    vec_cb = types.ModuleType("vector_service.context_builder")

    class ContextBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            return "", "", []

    vec_cb.ContextBuilder = ContextBuilder
    vec_cb.Retriever = object

    class FallbackResult(str):
        pass

    vec_cb.FallbackResult = FallbackResult

    class EmbeddingBackfill:
        def run(self, *a, **k):
            pass

    vec_cb.EmbeddingBackfill = EmbeddingBackfill
    sys.modules["vector_service.context_builder"] = vec_cb

    vec_pkg = types.ModuleType("vector_service")

    class ErrorResult(Exception):
        pass

    vec_pkg.ErrorResult = ErrorResult
    sys.modules["vector_service"] = vec_pkg

    chunk = types.ModuleType("chunking")
    chunk.get_chunk_summaries = lambda *a, **k: []
    sys.modules["chunking"] = chunk

    cdc = types.ModuleType("menace_sandbox.codebase_diff_checker")
    cdc.generate_code_diff = lambda *a, **k: {}
    cdc.flag_risky_changes = lambda *a, **k: []
    sys.modules["menace_sandbox.codebase_diff_checker"] = cdc

    haf = types.ModuleType("menace_sandbox.human_alignment_flagger")
    haf._collect_diff_data = lambda *a, **k: {}
    sys.modules["menace_sandbox.human_alignment_flagger"] = haf

    haa = types.ModuleType("menace_sandbox.human_alignment_agent")
    haa.HumanAlignmentAgent = object
    sys.modules["menace_sandbox.human_alignment_agent"] = haa

    vl = types.ModuleType("menace_sandbox.violation_logger")
    vl.log_violation = lambda *a, **k: None
    sys.modules["menace_sandbox.violation_logger"] = vl

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.quick_fix_engine", ROOT / "quick_fix_engine.py"
    )
    qfe = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.quick_fix_engine"] = qfe
    spec.loader.exec_module(qfe)

    class DummyDataBot:
        def __init__(self):
            self.roi_called = False
            self.errors_called = False
            self.db = types.SimpleNamespace(log_eval=lambda *a, **k: None)

        def roi(self, _name):
            self.roi_called = True
            return 0.0

        def average_errors(self, _name):
            self.errors_called = True
            return 0.0

    class DummyRegistry:
        def __init__(self):
            self.updated = None

        def register_bot(self, _name):
            pass

        def update_bot(self, name, module, **extra):
            self.updated = (name, module, extra)

    class DummyEngine:
        def generate_helper(self, desc, **kwargs):
            return "helper"

        def apply_patch_with_retry(self, path, helper, **kwargs):
            return 123, "", ""

    class DummyManager:
        def __init__(self):
            self.engine = DummyEngine()
            self.data_bot = DummyDataBot()
            self.bot_registry = DummyRegistry()
            self.bot_name = "dummy"
            self.cycle = None

        def register_patch_cycle(self, description, context_meta=None):
            self.cycle = (description, context_meta)
            self.data_bot.roi(self.bot_name)
            self.data_bot.average_errors(self.bot_name)

    manager = DummyManager()
    builder = ContextBuilder()

    mod = tmp_path / "mod.py"
    mod.write_text("print('hi')\n")

    commit = "deadbeef"
    monkeypatch.setattr(qfe.subprocess, "check_output", lambda *a, **k: commit.encode())

    patch_id = qfe.generate_patch(
        str(mod), manager, manager.engine, context_builder=builder
    )

    assert patch_id == 123
    assert manager.cycle is not None
    assert manager.data_bot.roi_called and manager.data_bot.errors_called
    assert manager.bot_registry.updated == (
        manager.bot_name,
        str(mod),
        {"patch_id": 123, "commit": commit},
    )
