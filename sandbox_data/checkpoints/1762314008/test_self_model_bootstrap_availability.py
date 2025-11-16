import types
import sys
import builtins
import pytest


def test_self_model_bootstrap_returns_identifier(monkeypatch):
    # Stub dependent modules before import
    dep_mod = types.ModuleType("menace.deployment_bot")

    class DummyDeploymentBot:
        def _record_workflows(self, *a, **k):
            return []

        def _update_bot_records(self, *a, **k):
            return {}

        def _record_code_templates(self, *a, **k):
            return None
    dep_mod.DeploymentBot = DummyDeploymentBot
    monkeypatch.setitem(sys.modules, "menace.deployment_bot", dep_mod)

    err_mod = types.ModuleType("menace.error_bot")

    class DummyErrorBot:
        def __init__(self, *a, **k):
            return None

        def record_runtime_error(self, *a, **k):
            return None

        def monitor(self):
            return None
    err_mod.ErrorBot = DummyErrorBot
    monkeypatch.setitem(sys.modules, "menace.error_bot", err_mod)

    data_mod = types.ModuleType("menace.data_bot")

    class DummyDB:
        def fetch(self, _):
            return types.SimpleNamespace(empty=True)

    class DummyDataBot:
        def __init__(self, *a, **k):
            self.db = DummyDB()

        def collect(self, *a, **k):
            return None
    data_mod.DataBot = DummyDataBot
    data_mod.MetricsDB = object  # placeholder
    monkeypatch.setitem(sys.modules, "menace.data_bot", data_mod)

    cap_mod = types.ModuleType("menace.capital_management_bot")

    class DummyCapital:
        def update_rois(self):
            return None
    cap_mod.CapitalManagementBot = DummyCapital
    monkeypatch.setitem(sys.modules, "menace.capital_management_bot", cap_mod)

    dbm_mod = types.ModuleType("menace.database_manager")
    dbm_mod.add_model = lambda *a, **k: 7
    dbm_mod.update_model = lambda *a, **k: None
    dbm_mod.DB_PATH = ""
    monkeypatch.setitem(sys.modules, "menace.database_manager", dbm_mod)

    cb_mod = types.ModuleType("vector_service.context_builder")

    class DummyBuilder:
        def refresh_db_weights(self):
            return None

    cb_mod.ContextBuilder = DummyBuilder
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", cb_mod)

    sys.modules.pop("menace.self_model_bootstrap", None)

    import menace.self_model_bootstrap as smb
    monkeypatch.setattr(smb.Path, "glob", lambda self, pattern: [])
    assert smb.bootstrap(context_builder=DummyBuilder()) == 7


def test_bootstrap_missing_module_raises(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "menace.self_model_bootstrap":
            raise ModuleNotFoundError("missing")
        return orig_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = (
        "try:\n"
        "    from menace.self_model_bootstrap import bootstrap\n"
        "except Exception:\n"
        "    def bootstrap(*_a, **_k) -> int:\n"
        "        raise RuntimeError('self_model_bootstrap module is required for bootstrapping')\n"
    )
    mod = types.ModuleType("tmp")
    exec(code, mod.__dict__)
    with pytest.raises(RuntimeError, match="self_model_bootstrap module is required"):
        mod.bootstrap()
