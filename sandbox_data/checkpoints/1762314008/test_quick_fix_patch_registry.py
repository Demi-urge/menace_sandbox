import sys
import types
import importlib.util
import importlib
from pathlib import Path
import concurrent.futures
import threading
import time


def _setup_env(tmp_path, monkeypatch):
    ROOT = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT))

    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(ROOT)]
    sys.modules["menace_sandbox"] = package

    dpr = types.ModuleType("dynamic_path_router")
    dpr.resolve_path = lambda p: Path(p)
    dpr.path_for_prompt = lambda p: str(p)
    dpr.resolve_dir = lambda p: Path(p)
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

    def record_patch_metadata(*a, **k):  # pragma: no cover - record calls
        record_patch_metadata.calls.append((a, k))

    record_patch_metadata.calls = []

    pp.PatchLogger = PatchLogger
    pp.record_patch_metadata = record_patch_metadata
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

    db_mod = types.ModuleType("menace_sandbox.data_bot")

    class DataBot:  # pragma: no cover - stub
        def __init__(self, *a, **k):
            pass

    db_mod.DataBot = DataBot
    sys.modules["menace_sandbox.data_bot"] = db_mod

    aem = types.ModuleType("menace_sandbox.advanced_error_management")

    class AutomatedRollbackManager:  # pragma: no cover - stub
        def __init__(self, *a, **k):
            self.rolled_back = None

        def rollback(self, patch_id):  # pragma: no cover - record call
            self.rolled_back = patch_id

    aem.AutomatedRollbackManager = AutomatedRollbackManager
    sys.modules["menace_sandbox.advanced_error_management"] = aem

    code_db = types.ModuleType("menace_sandbox.code_database")

    class PatchHistoryDB:  # pragma: no cover - stub
        def __init__(self, *a, **k):
            self.logged = []

        def record_vector_metrics(self, *a, **k):
            self.logged.append((a, k))

    code_db.PatchHistoryDB = PatchHistoryDB
    sys.modules["menace_sandbox.code_database"] = code_db
    sys.modules["code_database"] = code_db

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.quick_fix_engine", ROOT / "quick_fix_engine.py"
    )
    qfe = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.quick_fix_engine"] = qfe
    spec.loader.exec_module(qfe)
    return ROOT, qfe, ContextBuilder


def test_quick_fix_registers_registry_and_metrics(tmp_path, monkeypatch):
    ROOT, qfe, ContextBuilder = _setup_env(tmp_path, monkeypatch)
    pp_mod = importlib.import_module("patch_provenance")

    class Bus:
        def __init__(self):
            self.events = []

        def publish(self, name, payload):  # pragma: no cover - record events
            self.events.append((name, payload))

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

    scm_cls = importlib.import_module("menace_sandbox.self_coding_manager").SelfCodingManager

    class DummyManager(scm_cls):
        def __init__(self):
            self.engine = DummyEngine()
            self.data_bot = DummyDataBot()
            self.bot_registry = DummyRegistry()
            self.bot_name = "dummy"
            self.event_bus = Bus()
            self.cycle = None

        def register_patch_cycle(self, description, context_meta=None, provenance_token=None):
            self.cycle = (description, context_meta)
            self.data_bot.roi(self.bot_name)
            self.data_bot.average_errors(self.bot_name)

        def run_patch(self, path, desc, *, provenance_token, context_meta=None, context_builder=None):
            pid, _, _ = self.engine.apply_patch_with_retry(path, "helper")
            commit_hash = commit
            self.bot_registry.update_bot(
                self.bot_name, str(path), patch_id=pid, commit=commit_hash
            )
            pp_mod.record_patch_metadata(pid, {"commit": commit_hash, "module": str(path)})
            self.event_bus.publish(
                "bot:updated",
                {
                    "bot": self.bot_name,
                    "module": str(path),
                    "patch_id": pid,
                    "commit": commit_hash,
                },
            )
            return types.SimpleNamespace(patch_id=pid)

        def validate_provenance(self, _token):  # pragma: no cover - stub
            return True

    manager = DummyManager()
    builder = ContextBuilder()

    mod = tmp_path / "mod.py"
    mod.write_text("print('hi')\n")

    commit = "deadbeef"

    manager.register_patch_cycle("desc", provenance_token="tok")

    result = manager.run_patch(
        mod,
        "desc",
        provenance_token="tok",
        context_builder=builder,
    )
    patch_id = result.patch_id

    assert patch_id == 123
    assert manager.cycle is not None
    assert manager.data_bot.roi_called and manager.data_bot.errors_called
    assert manager.bot_registry.updated == (
        manager.bot_name,
        str(mod),
        {"patch_id": 123, "commit": commit},
    )
    assert (
        "bot:updated",
        {
            "bot": manager.bot_name,
            "module": str(mod),
            "patch_id": 123,
            "commit": commit,
        },
    ) in manager.event_bus.events
    assert pp_mod.record_patch_metadata.calls


def test_quick_fix_registry_updates_atomic(tmp_path, monkeypatch):
    ROOT, qfe, ContextBuilder = _setup_env(tmp_path, monkeypatch)
    bot_registry_mod = importlib.import_module("menace_sandbox.bot_registry")
    BotRegistry = bot_registry_mod.BotRegistry
    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", lambda *a, **k: True)
    monkeypatch.setattr(BotRegistry, "hot_swap_bot", lambda *a, **k: None)
    monkeypatch.setattr(BotRegistry, "health_check_bot", lambda *a, **k: None)

    class DummyDataBot:
        def __init__(self):
            self.db = types.SimpleNamespace(log_eval=lambda *a, **k: None)

        def roi(self, _name):
            return 0.0

        def average_errors(self, _name):
            return 0.0

    class DummyEngine:
        def __init__(self):
            self._counter = 0
            self._lock = threading.Lock()

        def generate_helper(self, desc, **kwargs):
            return "helper"

        def apply_patch_with_retry(self, path, helper, **kwargs):
            with self._lock:
                self._counter += 1
                pid = self._counter
            time.sleep(0.01)
            return pid, "", ""

    scm_cls = importlib.import_module("menace_sandbox.self_coding_manager").SelfCodingManager

    class DummyManager(scm_cls):
        def __init__(self, registry):
            self.engine = DummyEngine()
            self.data_bot = DummyDataBot()
            self.bot_registry = registry
            self.bot_name = "dummy"

        def register_patch_cycle(self, description, context_meta=None, provenance_token=None):
            pass

        def validate_provenance(self, _token):  # pragma: no cover - stub
            return True

    registry = BotRegistry()
    manager = DummyManager(registry)
    builder = ContextBuilder()

    mod1 = tmp_path / "mod1.py"
    mod2 = tmp_path / "mod2.py"
    mod1.write_text("print('hi')\n")
    mod2.write_text("print('hi')\n")

    commit = "deadbeef"
    monkeypatch.setattr(qfe.subprocess, "check_output", lambda *a, **k: commit.encode())

    def run(p):
        qfe.generate_patch(
            str(p),
            manager,
            manager.engine,
            context_builder=builder,
            provenance_token="tok",
            helper_fn=lambda *a, **k: "helper",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, [mod1, mod2]))

    node = registry.graph.nodes[manager.bot_name]
    assert registry.modules[manager.bot_name] == node["module"]


def test_quick_fix_runs_approval_policy(tmp_path, monkeypatch):
    ROOT, qfe, ContextBuilder = _setup_env(tmp_path, monkeypatch)

    class DummyDataBot:
        def roi(self, _name):
            return 0.0

        def average_errors(self, _name):
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
            return 1, "", ""

    class DummyApprovalPolicy:
        def __init__(self):
            self.called = False

        def approve(self, _path):
            self.called = True
            return True

    scm_cls = importlib.import_module("menace_sandbox.self_coding_manager").SelfCodingManager

    class DummyManager(scm_cls):
        def __init__(self):
            self.engine = DummyEngine()
            self.data_bot = DummyDataBot()
            self.bot_registry = DummyRegistry()
            self.bot_name = "dummy"
            self.approval_policy = DummyApprovalPolicy()

        def register_patch_cycle(self, description, context_meta=None, provenance_token=None):
            pass

        def validate_provenance(self, _token):  # pragma: no cover - stub
            return True

    manager = DummyManager()
    builder = ContextBuilder()
    mod = tmp_path / "mod.py"
    mod.write_text("print('hi')\n")
    commit = "deadbeef"
    monkeypatch.setattr(qfe.subprocess, "check_output", lambda *a, **k: commit.encode())

    pid = qfe.generate_patch(
        str(mod),
        manager,
        manager.engine,
        context_builder=builder,
        provenance_token="tok",
        helper_fn=lambda *a, **k: "helper",
    )

    assert pid == 1
    assert manager.approval_policy.called
    assert manager.bot_registry.updated == (
        manager.bot_name,
        str(mod),
        {"patch_id": 1, "commit": commit},
    )


def test_quick_fix_approval_failure_rolls_back_and_notifies(tmp_path, monkeypatch):
    ROOT, qfe, ContextBuilder = _setup_env(tmp_path, monkeypatch)
    aem = importlib.import_module("menace_sandbox.advanced_error_management")
    code_db = importlib.import_module("menace_sandbox.code_database")
    rb = aem.AutomatedRollbackManager()
    phdb = code_db.PatchHistoryDB()
    monkeypatch.setattr(qfe, "PatchHistoryDB", lambda *a, **k: phdb)

    class Bus:
        def __init__(self):
            self.events = []

        def publish(self, name, payload):  # pragma: no cover - record events
            self.events.append((name, payload))

    bus = Bus()

    class DummyDataBot:
        def roi(self, _name):
            return 0.0

        def average_errors(self, _name):
            return 0.0

    class DummyRegistry:
        def register_bot(self, _name):
            pass

        def update_bot(self, *a, **k):
            raise AssertionError("should not update registry")

    class DummyEngine:
        def generate_helper(self, desc, **kwargs):
            return "helper"

        def apply_patch_with_retry(self, path, helper, **kwargs):
            return 2, "", ""

    class DummyApprovalPolicy:
        rollback_mgr = rb

        def approve(self, _path):
            return False

    scm_cls = importlib.import_module("menace_sandbox.self_coding_manager").SelfCodingManager

    class DummyManager(scm_cls):
        def __init__(self):
            self.engine = DummyEngine()
            self.data_bot = DummyDataBot()
            self.bot_registry = DummyRegistry()
            self.bot_name = "dummy"
            self.approval_policy = DummyApprovalPolicy()
            self.event_bus = bus

        def register_patch_cycle(self, description, context_meta=None, provenance_token=None):
            pass

        def validate_provenance(self, _token):  # pragma: no cover - stub
            return True

    manager = DummyManager()
    builder = ContextBuilder()
    mod = tmp_path / "mod.py"
    mod.write_text("print('hi')\n")
    commit = "deadbeef"
    monkeypatch.setattr(qfe.subprocess, "check_output", lambda *a, **k: commit.encode())

    result = qfe.generate_patch(
        str(mod),
        manager,
        manager.engine,
        context_builder=builder,
        provenance_token="tok",
        helper_fn=lambda *a, **k: "helper",
    )

    assert result is None
    assert rb.rolled_back == "2"
    assert phdb.logged
    assert bus.events and bus.events[0][0] == "quick_fix:approval_failed"
