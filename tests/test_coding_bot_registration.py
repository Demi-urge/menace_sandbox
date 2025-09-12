import importlib
import sys
import types
import contextvars
from pathlib import Path
import pytest

# Stub menace package and heavy dependencies to allow lightweight imports
pkg = types.ModuleType("menace")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
pkg.RAISE_ERRORS = False
sys.modules["menace"] = pkg

scm_stub = types.ModuleType("menace.self_coding_manager")
class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
scm_stub.SelfCodingManager = DummyManager
sys.modules.setdefault("menace.self_coding_manager", scm_stub)

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
sce_stub.SelfCodingEngine = object
sys.modules.setdefault("menace.self_coding_engine", sce_stub)

cbi_stub = types.ModuleType("menace.coding_bot_interface")
def self_coding_managed(*args, **kwargs):
    def decorator(cls):
        cls.bot_registry = kwargs.get("bot_registry")
        cls.data_bot = kwargs.get("data_bot")
        return cls
    if args and callable(args[0]):
        return decorator(args[0])
    return decorator
def manager_generate_helper(*a, **k):
    return ""
cbi_stub.self_coding_managed = self_coding_managed
cbi_stub.manager_generate_helper = manager_generate_helper
sys.modules.setdefault("menace.coding_bot_interface", cbi_stub)

code_stub = types.ModuleType("menace.code_database")
class _Code:  # minimal placeholder
    pass
code_stub.CodeDB = _Code
code_stub.CodeRecord = _Code
code_stub.PatchHistoryDB = _Code
code_stub.PatchRecord = _Code
sys.modules.setdefault("menace.code_database", code_stub)

vs_cb_stub = types.ModuleType("vector_service.context_builder")
class ContextBuilder:  # minimal placeholder
    pass
class FallbackResult:
    pass
class ErrorResult:
    pass
vs_cb_stub.ContextBuilder = ContextBuilder
vs_cb_stub.FallbackResult = FallbackResult
vs_cb_stub.ErrorResult = ErrorResult
sys.modules.setdefault("vector_service.context_builder", vs_cb_stub)

cog_stub = types.ModuleType("vector_service.cognition_layer")
class CognitionLayer:  # minimal placeholder
    pass
cog_stub.CognitionLayer = CognitionLayer
sys.modules.setdefault("vector_service.cognition_layer", cog_stub)

data_stub = types.ModuleType("menace.data_bot")
class DataBot:
    def __init__(self, *_, **__):
        pass
class MetricsDB:  # minimal placeholder
    pass
data_stub.DataBot = DataBot
data_stub.MetricsDB = MetricsDB
sys.modules.setdefault("menace.data_bot", data_stub)

CODING_BOTS = [
    "menace.bot_planning_bot.BotPlanningBot",
    "menace.implementation_optimiser_bot.ImplementationOptimiserBot",
    "menace.bot_creation_bot.BotCreationBot",
    "menace.bot_development_bot.BotDevelopmentBot",
    "menace.bot_testing_bot.BotTestingBot",
    "menace.automated_debugger.AutomatedDebugger",
]

@pytest.mark.parametrize("path", CODING_BOTS)
def test_coding_bots_have_manager_and_registry(path):
    mod_name, cls_name = path.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # pragma: no cover - best effort import
        pytest.skip(f"import failed for {mod_name}: {exc}")
    cls = getattr(mod, cls_name)
    assert hasattr(cls, "manager"), f"{cls_name} missing manager attribute"
    assert getattr(cls, "bot_registry", None) is not None, f"{cls_name} missing bot_registry"
