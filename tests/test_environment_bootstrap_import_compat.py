import importlib
import subprocess
import sys
import types


def _install_stub_bootstrap_module():
    module = types.ModuleType("menace_sandbox.environment_bootstrap")

    class EnvironmentBootstrapper:
        pass

    module.EnvironmentBootstrapper = EnvironmentBootstrapper
    module.BootstrapOrchestrator = type("BootstrapOrchestrator", (), {})
    module.ensure_bootstrapped = lambda *args, **kwargs: None
    module.ensure_bootstrapped_async = lambda *args, **kwargs: None
    sys.modules["menace_sandbox.environment_bootstrap"] = module


def test_environment_bootstrap_module_exposes_bootstrapper(monkeypatch):
    _install_stub_bootstrap_module()
    monkeypatch.delitem(sys.modules, "menace.environment_bootstrap", raising=False)

    module = importlib.import_module("menace.environment_bootstrap")

    assert hasattr(module, "EnvironmentBootstrapper")


def test_direct_import_environment_bootstrapper_is_type(monkeypatch):
    _install_stub_bootstrap_module()
    monkeypatch.delitem(sys.modules, "menace.environment_bootstrap", raising=False)

    from menace.environment_bootstrap import EnvironmentBootstrapper

    assert EnvironmentBootstrapper is not None
    assert isinstance(EnvironmentBootstrapper, type)


def test_python_c_import_environment_bootstrapper():
    script = (
        "import sys, types;"
        "mod=types.ModuleType('menace_sandbox.environment_bootstrap');"
        "mod.EnvironmentBootstrapper=type('EnvironmentBootstrapper', (), {});"
        "mod.BootstrapOrchestrator=type('BootstrapOrchestrator', (), {});"
        "mod.ensure_bootstrapped=lambda *a, **k: None;"
        "mod.ensure_bootstrapped_async=lambda *a, **k: None;"
        "sys.modules['menace_sandbox.environment_bootstrap']=mod;"
        "from menace.environment_bootstrap import EnvironmentBootstrapper;"
        "assert EnvironmentBootstrapper is not None;"
        "assert isinstance(EnvironmentBootstrapper, type)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
