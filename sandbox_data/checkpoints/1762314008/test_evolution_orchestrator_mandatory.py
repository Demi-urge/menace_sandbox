import sys
import types
import pytest

vcb = types.ModuleType("vector_service.context_builder")


class _StubBuilder:
    def refresh_db_weights(self) -> None:
        pass


vcb.ContextBuilder = _StubBuilder
vcb.record_failed_tags = lambda *a, **k: None
vcb.load_failed_tags = lambda *a, **k: []
sys.modules.setdefault("vector_service", types.ModuleType("vector_service"))
sys.modules["vector_service.context_builder"] = vcb

ffs = types.ModuleType("menace.failure_fingerprint_store")
ffs.FailureFingerprint = object
ffs.FailureFingerprintStore = object
sys.modules["menace.failure_fingerprint_store"] = ffs
sys.modules["failure_fingerprint_store"] = ffs

ff = types.ModuleType("menace.failure_fingerprint")
ff.FailureFingerprint = object
sys.modules["menace.failure_fingerprint"] = ff
sys.modules["failure_fingerprint"] = ff

fru = types.ModuleType("menace.failure_retry_utils")
fru.check_similarity_and_warn = lambda *a, **k: None
fru.record_failure = lambda *a, **k: None
sys.modules["menace.failure_retry_utils"] = fru
sys.modules["failure_retry_utils"] = fru

sr_pkg = types.ModuleType("menace.sandbox_runner")
th = types.ModuleType("menace.sandbox_runner.test_harness")
th.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th.TestHarnessResult = types.SimpleNamespace
sr_pkg.test_harness = th
sys.modules["menace.sandbox_runner"] = sr_pkg
sys.modules["menace.sandbox_runner.test_harness"] = th
sys.modules["sandbox_runner.test_harness"] = th

stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)

data_bot_stub = types.ModuleType("menace.data_bot")


class _DataBot:
    def __init__(self, *a, **k):
        self.db = types.SimpleNamespace()


data_bot_stub.DataBot = _DataBot
data_bot_stub.MetricsDB = object
sys.modules["menace.data_bot"] = data_bot_stub
sys.modules["data_bot"] = data_bot_stub

db_router_stub = types.ModuleType("db_router")
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.LOCAL_TABLES = set()


class _DummyRouter:
    menace_id = "test"

    def __init__(self, *a, **k):
        pass

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *a, **k):
            return types.SimpleNamespace(fetchall=lambda: [])

        def commit(self):
            pass

    def get_connection(self, *_a, **_k):
        return self._Conn()


def init_db_router(*a, **k):
    return _DummyRouter()


db_router_stub.DBRouter = _DummyRouter
db_router_stub.init_db_router = init_db_router
sys.modules.setdefault("db_router", db_router_stub)

dpr = types.SimpleNamespace(
    resolve_path=lambda p: __import__("pathlib").Path(p),
    repo_root=lambda: __import__("pathlib").Path("."),
    path_for_prompt=lambda p: str(p),
    get_project_root=lambda: __import__("pathlib").Path("."),
)
sys.modules["dynamic_path_router"] = dpr

mapl_stub = types.ModuleType("menace.model_automation_pipeline")


class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi


class ModelAutomationPipeline:
    ...


mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sce_stub.MANAGER_CONTEXT = {}
sys.modules["menace.self_coding_engine"] = sce_stub

qfe_stub = types.ModuleType("menace.quick_fix_engine")
qfe_stub.QuickFixEngine = object
qfe_stub.generate_patch = lambda *a, **k: (1, [])
sys.modules["menace.quick_fix_engine"] = qfe_stub
sys.modules["quick_fix_engine"] = qfe_stub

error_bot_stub = types.ModuleType("menace.error_bot")
error_bot_stub.ErrorDB = object
sys.modules["menace.error_bot"] = error_bot_stub

aem_stub = types.ModuleType("menace.advanced_error_management")
aem_stub.FormalVerifier = object
aem_stub.AutomatedRollbackManager = object
sys.modules["menace.advanced_error_management"] = aem_stub

rm_stub = types.ModuleType("menace.rollback_manager")
rm_stub.RollbackManager = object
sys.modules["menace.rollback_manager"] = rm_stub

mutation_logger_stub = types.ModuleType("menace.mutation_logger")
mutation_logger_stub.log_mutation = lambda *a, **k: None
sys.modules["menace.mutation_logger"] = mutation_logger_stub

code_db_stub = types.ModuleType("menace.code_database")


class PatchRecord:
    pass


class PatchHistoryDB:
    pass


code_db_stub.PatchRecord = PatchRecord
code_db_stub.PatchHistoryDB = PatchHistoryDB
sys.modules["menace.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub

psdb = types.ModuleType("patch_suggestion_db")
psdb.PatchSuggestionDB = object
sys.modules["patch_suggestion_db"] = psdb

psafety = types.ModuleType("patch_safety")
psafety.PatchSafety = object
sys.modules["patch_safety"] = psafety

ev_stub = types.ModuleType("error_vectorizer")
ev_stub.ErrorVectorizer = object
sys.modules["error_vectorizer"] = ev_stub

pp_stub = types.ModuleType("patch_provenance")
pp_stub.record_patch_metadata = lambda *a, **k: None
sys.modules["patch_provenance"] = pp_stub

sem_stub = types.ModuleType("menace.system_evolution_manager")


class _SEM:
    def __init__(self, bots):
        pass


sem_stub.SystemEvolutionManager = _SEM
sys.modules["menace.system_evolution_manager"] = sem_stub

si_mod = types.ModuleType("menace.self_improvement.engine")
si_mod.SelfImprovementEngine = object
sys.modules["menace.self_improvement.engine"] = si_mod
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.engine = si_mod
sys.modules["menace.self_improvement"] = si_pkg
si_baseline = types.ModuleType("menace.self_improvement.baseline_tracker")
si_baseline.BaselineTracker = lambda *a, **k: object()
sys.modules["menace.self_improvement.baseline_tracker"] = si_baseline
si_pkg.baseline_tracker = si_baseline
si_target = types.ModuleType("menace.self_improvement.target_region")
si_target.TargetRegion = object
sys.modules["menace.self_improvement.target_region"] = si_target
si_pkg.target_region = si_target
import menace
menace.self_improvement = si_pkg

import menace.self_coding_manager as scm


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyEngine:
    cognition_layer = types.SimpleNamespace(context_builder=DummyBuilder())


class DummyPipeline:
    pass


class DummyDataBot:
    pass


class DummyRegistry:
    graph = {}


class DummyQuickFix:
    pass


def test_init_registers_with_orchestrator():
    class DummyOrchestrator:
        def __init__(self):
            self.registered = []

        def register_bot(self, name: str) -> None:
            self.registered.append(name)

    orch = DummyOrchestrator()
    scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=DummyQuickFix(),
        evolution_orchestrator=orch,
    )
    assert orch.registered == ["bot"]


def test_init_raises_when_orchestrator_fails(monkeypatch):
    monkeypatch.setattr(
        "menace.capital_management_bot.CapitalManagementBot",
        lambda data_bot=None: object(),
    )
    monkeypatch.setattr(
        "menace.self_improvement.engine.SelfImprovementEngine",
        lambda data_bot=None, bot_name=None: object(),
    )

    def fail(*_a, **_k):
        raise Exception("boom")

    monkeypatch.setattr(
        "menace.evolution_orchestrator.EvolutionOrchestrator", fail
    )

    with pytest.raises(RuntimeError, match="EvolutionOrchestrator"):
        scm.SelfCodingManager(
            DummyEngine(),
            DummyPipeline(),
            bot_name="bot",
            data_bot=DummyDataBot(),
            bot_registry=DummyRegistry(),
            quick_fix=DummyQuickFix(),
        )
