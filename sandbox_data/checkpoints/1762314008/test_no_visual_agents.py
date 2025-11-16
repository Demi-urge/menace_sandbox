import os
import sys
import importlib
from types import ModuleType
from pathlib import Path
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Ensure menace package exists with RAISE_ERRORS flag
menace_pkg = sys.modules.get("menace")
if menace_pkg is None:
    menace_pkg = ModuleType("menace")
    menace_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["menace"] = menace_pkg
menace_pkg.RAISE_ERRORS = False

# Lightweight stubs for heavy dependencies
stub = ModuleType("db_router")
stub.DBRouter = object
stub.GLOBAL_ROUTER = None
stub.init_db_router = lambda *a, **k: None
stub.LOCAL_TABLES = {}
stub.SHARED_TABLES = {}
stub.queue_insert = lambda *a, **k: None
sys.modules.setdefault("db_router", stub)
sys.modules.setdefault("menace.db_router", stub)

vec_stub = ModuleType("vector_service")


class _CB:
    def __init__(self, *a, **k):
        pass

    def build(self, *a, **k):
        return ""

    def refresh_db_weights(self):
        pass


vec_stub.ContextBuilder = _CB


class _FallbackResult:
    def __init__(self, *a, **k):
        pass


vec_stub.FallbackResult = _FallbackResult
vec_stub.ErrorResult = type("ErrorResult", (), {})
sys.modules.setdefault("vector_service", vec_stub)
sys.modules.setdefault("vector_service.context_builder", vec_stub)

sc_stub = ModuleType("snippet_compressor")
sc_stub.compress_snippets = lambda meta, **k: meta
sys.modules.setdefault("snippet_compressor", sc_stub)

code_db_stub = ModuleType("code_database")
code_db_stub.CodeDB = object
sys.modules["code_database"] = code_db_stub
sys.modules["menace.code_database"] = code_db_stub

mm_stub = ModuleType("menace_memory_manager")


class _MM:
    pass


mm_stub.MenaceMemoryManager = _MM
sys.modules["menace_memory_manager"] = mm_stub
sys.modules["menace.menace_memory_manager"] = mm_stub

sce_stub = ModuleType("self_coding_engine")


class _DummyEngine:
    def __init__(self, *a, **k):
        pass


sce_stub.SelfCodingEngine = _DummyEngine
sys.modules.setdefault("self_coding_engine", sce_stub)
sys.modules.setdefault("menace.self_coding_engine", sce_stub)

# Load menace package modules
pkg_path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace", pkg_path / "__init__.py", submodule_search_locations=[str(pkg_path)]
)
menace_mod = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace_mod
spec.loader.exec_module(menace_mod)
menace_mod.RAISE_ERRORS = False
bdb = importlib.import_module("menace.bot_development_bot")
cfg_mod = importlib.import_module("menace.bot_dev_config")


def test_no_visual_agents(tmp_path):
    cfg = cfg_mod.BotDevConfig()
    cfg.visual_agents = ["demo"]  # type: ignore[attr-defined]
    ctx = vec_stub.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    engine = _DummyEngine()
    assert "vision_utils" not in sys.modules
    with pytest.raises(ValueError):
        bdb.BotDevelopmentBot(repo_base=tmp_path, context_builder=ctx, engine=engine, config=cfg)
    assert "vision_utils" not in sys.modules
