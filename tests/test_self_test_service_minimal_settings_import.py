from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _DummyBuilder:
    def refresh_db_weights(self) -> None:
        return None

    def build_context(self, *_a, **_k):
        return "ctx", "summary", {}


def _load_self_test_service_with_minimal_settings() -> types.ModuleType:
    for name in ["menace", "menace.self_test_service", "self_test_service", "sandbox_settings"]:
        sys.modules.pop(name, None)

    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
    sys.modules["menace"] = pkg

    filelock_mod = types.ModuleType("filelock")

    pydantic_compat = types.ModuleType("pydantic_settings_compat")

    class _BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic_compat.BaseSettings = _BaseSettings
    sys.modules["pydantic_settings_compat"] = pydantic_compat

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.Field = lambda default=None, **_kwargs: default
    pydantic_mod.ValidationError = Exception
    sys.modules["pydantic"] = pydantic_mod


    class _DummyFileLock:
        def __init__(self, *_a, **_k) -> None:
            pass

    filelock_mod.FileLock = _DummyFileLock
    sys.modules["filelock"] = filelock_mod

    sandbox_settings = types.ModuleType("sandbox_settings")
    sandbox_settings.SandboxSettings = lambda: types.SimpleNamespace()
    sys.modules["sandbox_settings"] = sandbox_settings
    sys.modules["menace.sandbox_settings"] = sandbox_settings

    vector_mod = types.ModuleType("vector_service.context_builder")
    vector_mod.ContextBuilder = _DummyBuilder
    sys.modules["vector_service.context_builder"] = vector_mod

    log_mod = types.ModuleType("logging_utils")
    log_mod.setup_logging = lambda: None
    log_mod.log_record = lambda **_k: {}
    log_mod.get_logger = lambda *_a, **_k: None
    sys.modules["logging_utils"] = log_mod
    sys.modules["menace.logging_utils"] = log_mod

    db_router_mod = types.ModuleType("db_router")
    db_router_mod.init_db_router = lambda *_a, **_k: types.SimpleNamespace(get_connection=lambda *_a, **_k: None)
    db_router_mod.DBRouter = object
    db_router_mod.GLOBAL_ROUTER = None
    db_router_mod.LOCAL_TABLES = {}
    sys.modules["db_router"] = db_router_mod


    self_services_mod = types.ModuleType("self_services_config")

    class _SelfTestConfig:
        def __init__(self) -> None:
            self.lock_file = ROOT / "sandbox_data" / "self_test.lock"
            self.report_dir = ROOT / "sandbox_data" / "self_test_reports"

    self_services_mod.SelfTestConfig = _SelfTestConfig
    sys.modules["self_services_config"] = self_services_mod
    sys.modules["menace.self_services_config"] = self_services_mod

    scoring_mod = types.ModuleType("sandbox_runner.scoring")
    scoring_mod.record_run = lambda *_a, **_k: None
    sys.modules["sandbox_runner.scoring"] = scoring_mod

    for name, attr in {
        "snippet_compressor": "compress_snippets",
        "orphan_analyzer": "classify_module",
    }.items():
        module = types.ModuleType(name)
        setattr(module, attr, lambda *_a, **_k: None)
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location("menace.self_test_service", ROOT / "self_test_service.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["menace.self_test_service"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_import_tolerates_minimal_sandbox_settings_stub():
    module = _load_self_test_service_with_minimal_settings()

    assert module._RECURSIVE_ORPHAN_SCAN is True
    assert module._AUTO_INCLUDE_ISOLATED is False
    assert module._RECURSIVE_ISOLATED is False
    assert module._TEST_REDUNDANT_MODULES is False
