import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Minimal stubs so self_coding_engine imports without heavy deps
package = types.ModuleType("menace_sandbox")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("menace_sandbox", package)


def _setmod(name: str, module: object) -> None:
    sys.modules[name] = module
    sys.modules[f"menace_sandbox.{name}"] = module


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
    record=lambda payload: None
)
_setmod("audit_trail", audit_mod)
access_mod = types.SimpleNamespace(
    READ="r", WRITE="w", check_permission=lambda *a, **k: None
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
        TestHarnessResult=types.SimpleNamespace(success=True, stdout=""),
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
        ),
        load_sandbox_settings=lambda: None,
    ),
)
roi_mod = types.ModuleType("roi_tracker")
roi_mod.ROITracker = lambda: object()
_setmod("roi_tracker", roi_mod)


class _BT:
    def __init__(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass


baseline_mod = types.SimpleNamespace(BaselineTracker=_BT, TRACKER=_BT())


class _FL:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


init_mod = types.SimpleNamespace(FileLock=_FL, _atomic_write=lambda *a, **k: None)
_setmod("self_improvement.baseline_tracker", baseline_mod)
_setmod("self_improvement.init", init_mod)
si_pkg = types.ModuleType("self_improvement")
si_pkg.__path__ = []
_setmod("self_improvement", si_pkg)


import menace_sandbox.self_coding_engine as sce  # noqa: E402
from chunking import CodeChunk  # noqa: E402
import ast  # noqa: E402


class DummyLLM:
    gpt_memory = None

    def generate(self, prompt):
        return types.SimpleNamespace(text="")


class Tracker:
    def __init__(self):
        self.calls = []

    def update(self, a, b):
        self.calls.append((a, b))


def _setup_engine(tmp_path, monkeypatch):
    engine = sce.SelfCodingEngine(
        object(),
        object(),
        llm_client=DummyLLM(),
        prompt_chunk_token_threshold=50,
        chunk_summary_cache_dir=tmp_path,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    engine.data_bot = None
    engine.trend_predictor = None
    engine.cognition_layer = None
    engine.patch_db = None
    engine.rollback_mgr = None
    engine.formal_verifier = None
    engine.roi_tracker = Tracker()
    return engine


def test_multi_chunk_patch_success(tmp_path, monkeypatch):
    engine = _setup_engine(tmp_path, monkeypatch)
    path = tmp_path / "big.py"  # path-ignore
    path.write_text("def a():\n    pass\n\ndef b():\n    pass\n")

    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)
    monkeypatch.setattr(
        sce,
        "split_into_chunks",
        lambda code, limit: [
            CodeChunk(1, 2, "def a():\n    pass", "h1", 5),
            CodeChunk(4, 5, "def b():\n    pass", "h2", 5),
        ],
    )
    import chunking as pc

    monkeypatch.setattr(
        pc, "summarize_code", lambda text, llm, context_builder=None: text.splitlines()[0]
    )

    calls = []

    def fake_generate_helper(desc, *a, **k):
        calls.append(desc)
        idx = len(calls)
        return f"# patch {idx}"

    monkeypatch.setattr(engine, "generate_helper", fake_generate_helper)
    monkeypatch.setattr(
        engine, "_run_ci", lambda p: types.SimpleNamespace(success=True)
    )

    engine.apply_patch(path, "add patches")

    lines = path.read_text().splitlines()
    assert "# patch 1" in lines
    assert "# patch 2" in lines
    assert len(calls) == 2
    assert len(engine.roi_tracker.calls) == 2


def test_multi_chunk_patch_with_rollback(tmp_path, monkeypatch):
    engine = _setup_engine(tmp_path, monkeypatch)
    path = tmp_path / "big.py"  # path-ignore
    path.write_text("def a():\n    pass\n\ndef b():\n    pass\n")

    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)
    monkeypatch.setattr(
        sce,
        "split_into_chunks",
        lambda code, limit: [
            CodeChunk(1, 2, "def a():\n    pass", "h1", 5),
            CodeChunk(4, 5, "def b():\n    pass", "h2", 5),
        ],
    )
    import chunking as pc

    monkeypatch.setattr(
        pc, "summarize_code", lambda text, llm, context_builder=None: text.splitlines()[0]
    )

    calls = []

    def fake_generate_helper(desc, *a, **k):
        calls.append(desc)
        idx = len(calls)
        return f"# patch {idx}"

    monkeypatch.setattr(engine, "generate_helper", fake_generate_helper)

    ci_calls = []

    def fake_run_ci(p):
        idx = len(ci_calls)
        ci_calls.append(idx)
        return types.SimpleNamespace(success=idx == 0)

    monkeypatch.setattr(engine, "_run_ci", fake_run_ci)

    engine.apply_patch(path, "add patches")

    lines = path.read_text().splitlines()
    assert "# patch 1" in lines
    assert "# patch 2" not in lines
    assert len(calls) == 2
    assert len(engine.roi_tracker.calls) == 1


def test_region_patch_indentation_and_ast(tmp_path, monkeypatch):
    engine = _setup_engine(tmp_path, monkeypatch)
    path = tmp_path / "f.py"  # path-ignore
    path.write_text("def a():\n    x = 1\n    y = 2\n    return x + y\n")

    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)
    monkeypatch.setattr(engine, "_run_ci", lambda p: types.SimpleNamespace(success=True))

    calls = []

    def fake_generate(desc, *a, **k):
        calls.append(k.get("target_region"))
        return "x=10\ny=20"

    monkeypatch.setattr(engine, "generate_helper", fake_generate)

    region = sce.TargetRegion(2, 3, "a")
    engine.apply_patch(path, "edit", target_region=region)

    src = path.read_text()
    assert "x=10" in src and "y=20" in src
    ast.parse(src)
    assert len(calls) == 1


def test_region_patch_fallback_on_parse_error(tmp_path, monkeypatch):
    engine = _setup_engine(tmp_path, monkeypatch)
    path = tmp_path / "f.py"  # path-ignore
    path.write_text("def a():\n    x = 1\n    return x\n")

    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)
    monkeypatch.setattr(engine, "_run_ci", lambda p: types.SimpleNamespace(success=True))

    outputs = ["x =", "def a():\n    return 5"]
    calls = []

    def fake_generate(desc, *a, **k):
        calls.append(k.get("target_region"))
        return outputs[len(calls) - 1]

    monkeypatch.setattr(engine, "generate_helper", fake_generate)

    region = sce.TargetRegion(2, 2, "a")
    engine.apply_patch(path, "edit", target_region=region)

    assert path.read_text() == "def a():\n    return 5\n"
    assert len(calls) == 2
