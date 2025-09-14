import sys
import types
from pathlib import Path

# Stub heavy dependencies before importing the orchestrator
cbi = types.ModuleType("menace.coding_bot_interface")
# accept arbitrary decorator arguments
cbi.self_coding_managed = lambda *a, **k: (lambda f: f)
sys.modules.setdefault("menace.coding_bot_interface", cbi)

# minimal dynamic path resolver
dpr = types.ModuleType("dynamic_path_router")
dpr.resolve_path = lambda p: Path(p)
sys.modules.setdefault("dynamic_path_router", dpr)

# system evolution manager stub
sem = types.ModuleType("menace.system_evolution_manager")
sem.SystemEvolutionManager = type("SystemEvolutionManager", (), {})
sys.modules.setdefault("menace.system_evolution_manager", sem)

# capital management stub to satisfy imports
cmb = types.ModuleType("menace.capital_management_bot")
cmb.CapitalManagementBot = type("CapitalManagementBot", (), {})
sys.modules.setdefault("menace.capital_management_bot", cmb)

# vector service context builder stub
vs_cb = types.ModuleType("vector_service.context_builder")
vs_cb.ContextBuilder = object
sys.modules.setdefault("vector_service.context_builder", vs_cb)

# mutation logger stub to avoid database setup
mut = types.ModuleType("menace.mutation_logger")
mut.log_mutation = lambda *a, **k: 1
mut.record_mutation_outcome = lambda *a, **k: None
class _Ctx:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False
mut.log_context = lambda *a, **k: _Ctx()
sys.modules.setdefault("menace.mutation_logger", mut)

# lightweight SelfCodingManager stub
scm_mod = types.ModuleType("menace.self_coding_manager")
class SelfCodingManager:
    def __init__(self, **kwargs):
        self.bot_name = kwargs.get("bot_name", "dummy")
        module = kwargs.get("module_path", Path("dummy.py"))
        name = self.bot_name
        class _Graph(dict):
            def __init__(self, module: Path):
                super().__init__({name: {"module": str(module)}})
            @property
            def nodes(self):
                return self
        self.bot_registry = types.SimpleNamespace(graph=_Graph(module))
        self.calls = []
    def register_patch_cycle(self, description, context_meta=None, provenance_token=None):
        self.calls.append((description, context_meta))
    def run_patch(self, *a, **k):
        pass
    def should_refactor(self) -> bool:
        return True
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = RuntimeError
sys.modules.setdefault("menace.self_coding_manager", scm_mod)

from menace.data_bot import DataBot, MetricsDB
from menace.evolution_orchestrator import EvolutionOrchestrator

# disable shared event bus to force fallback subscriptions
import menace.data_bot as db_mod
import menace.evolution_orchestrator as eo_mod
db_mod._SHARED_EVENT_BUS = None
eo_mod._SHARED_EVENT_BUS = None

def test_data_bot_threshold_breach_triggers_patch_cycle(tmp_path):
    db = MetricsDB(path=tmp_path / "metrics.db")
    data_bot = DataBot(db=db, event_bus=None)
    module = tmp_path / "dummy.py"
    module.write_text("def foo():\n    return 1\n")
    manager = SelfCodingManager(bot_name="dummy", module_path=module)
    EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(trend_predictor=None),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        history_db=types.SimpleNamespace(add=lambda *a, **k: None),
    )
    data_bot.check_degradation("dummy", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy", roi=0.0, errors=5.0)
    assert manager.calls
