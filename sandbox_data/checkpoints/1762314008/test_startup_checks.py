import sys
import logging
import types
import importlib.util
import importlib.machinery
from pathlib import Path
from typing import Sequence
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import base64
from dynamic_path_router import resolve_path

resolve_path(".")  # path-ignore

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"scpkg.{name}", ROOT / f"{name}.py"
    )  # path-ignore
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"scpkg.{name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pkg = types.ModuleType("scpkg")
pkg.__path__ = [str(ROOT)]
pkg.__spec__ = importlib.machinery.ModuleSpec("scpkg", loader=None, is_package=True)
sys.modules["scpkg"] = pkg

AuditTrail = _load("audit_trail").AuditTrail
_load("dependency_verifier")
sc = _load("startup_checks")


def test_validate_config_warns(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("MENACE_MODE", "test")
    for var in sc.REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)
    missing = sc.validate_config()
    assert set(missing) == set(sc.REQUIRED_VARS)
    assert "Missing configuration variables" in caplog.text


def test_validate_config_raises(monkeypatch):
    monkeypatch.setenv("MENACE_MODE", "production")
    for var in sc.REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        sc.validate_config()


def _write_pyproject(path, deps):
    path.write_text("""[project]\ndependencies = [\n""" +
                    "\n".join(f'    "{d}",' for d in deps) + "\n]\n")


def test_run_startup_checks_warns(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["fake_package_123"])
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    sc.run_startup_checks(pyproject_path=str(pyproj))
    assert "Missing required dependencies" in caplog.text


def test_optional_dependency_install(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])

    called: list[str] = []

    monkeypatch.setattr(
        sc, "validate_dependencies", lambda modules=sc.OPTIONAL_LIBS: ["missing_pkg"]
    )
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "_install_packages", lambda pkgs: called.extend(pkgs))
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)

    sc.run_startup_checks(pyproject_path=str(pyproj))

    assert "missing_pkg" in called


def test_run_startup_checks_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "production")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["fake_package_456"])
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        sc.run_startup_checks(pyproject_path=str(pyproj))


def test_verify_project_dependencies_accepts_str(monkeypatch, tmp_path):
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["pkg"])
    monkeypatch.setattr(sc, "verify_modules", lambda mods: mods)
    assert sc.verify_project_dependencies(str(pyproj)) == ["pkg"]


def test_dependency_checks_with_env_root(monkeypatch, tmp_path):
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["env_pkg"])
    monkeypatch.setenv("MENACE_ROOT", str(tmp_path))
    monkeypatch.setenv("MENACE_MODE", "test")

    orig_dpr = sys.modules.get("scpkg.dynamic_path_router")
    orig_sc = sys.modules.get("scpkg.startup_checks")
    try:
        _load("dynamic_path_router")
        sc_env = _load("startup_checks")

        seen: dict[str, list[str]] = {}

        def fake_verify(mods: Sequence[str]) -> list[str]:
            seen["mods"] = list(mods)
            return []

        monkeypatch.setattr(
            sc_env, "validate_dependencies", lambda modules=sc_env.OPTIONAL_LIBS: []
        )
        monkeypatch.setattr(sc_env, "verify_optional_dependencies", lambda: [])
        monkeypatch.setattr(sc_env, "validate_config", lambda: [])
        monkeypatch.setattr(sc_env, "verify_critical_libs", lambda: None)
        monkeypatch.setattr(sc_env, "verify_stripe_router", lambda *a, **k: None)
        monkeypatch.setattr(sc_env, "verify_modules", fake_verify)

        sc_env.run_startup_checks()
        assert seen["mods"] == ["env_pkg"]
    finally:
        if orig_dpr is not None:
            sys.modules["scpkg.dynamic_path_router"] = orig_dpr
        if orig_sc is not None:
            sys.modules["scpkg.startup_checks"] = orig_sc


@pytest.mark.skipif(not hasattr(Ed25519PrivateKey, "private_bytes"), reason="cryptography stubs")
def test_audit_log_verification(monkeypatch, tmp_path):
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])
    path = tmp_path / "audit.log"
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    trail = AuditTrail(str(path), priv_bytes)
    trail.record("msg")
    pub_b64 = base64.b64encode(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode()
    monkeypatch.setenv("AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("AUDIT_PUBKEY", pub_b64)
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    sc.run_startup_checks(pyproject_path=str(pyproj))
    # Corrupt the log
    with open(path, "a") as fh:
        fh.write("bad entry\n")
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        sc.run_startup_checks(pyproject_path=str(pyproj))


def test_run_startup_checks_invokes_optional_verifier(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])
    called = {"val": False}

    def fake_verify() -> list[str]:
        called["val"] = True
        return []

    monkeypatch.setattr(sc, "verify_optional_dependencies", fake_verify)
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)

    sc.run_startup_checks(pyproject_path=str(pyproj))

    assert called["val"]


def test_run_startup_checks_invokes_stripe_router(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])
    called = {"val": False}

    def fake_verify(*a, **k) -> None:
        called["val"] = True

    monkeypatch.setattr(sc, "verify_stripe_router", fake_verify)
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])

    sc.run_startup_checks(pyproject_path=str(pyproj))

    assert called["val"]


def test_run_startup_checks_skips_stripe_router(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])
    called = {"val": False}

    def fake_verify(*a, **k) -> None:
        called["val"] = True

    monkeypatch.setattr(sc, "verify_stripe_router", fake_verify)
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])

    sc.run_startup_checks(pyproject_path=str(pyproj), skip_stripe_router=True)

    assert not called["val"]


def test_skip_stripe_router_suppresses_optional_stripe_import(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])

    monkeypatch.setattr(sc, "validate_dependencies", lambda modules=sc.OPTIONAL_LIBS: [])
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "validate_config", lambda: [])
    monkeypatch.setattr(sc, "verify_critical_libs", lambda: None)
    monkeypatch.setattr(sc, "_prompt_for_vars", lambda names: None)
    monkeypatch.setattr(sc, "_install_packages", lambda pkgs: None)
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: pytest.fail("stripe router should be skipped"))

    captured: dict[str, list[str] | None] = {}
    original_optional = sc.verify_optional_dependencies

    def fake_optional(modules=None):
        captured["modules"] = list(modules) if modules is not None else None
        return original_optional(modules)

    monkeypatch.setattr(sc, "verify_optional_dependencies", fake_optional)

    original_import = sc.importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if "stripe_billing_router" in name:
            raise AssertionError("stripe_billing_router import attempted")
        if name == "quick_fix_engine":
            raise AssertionError("quick_fix_engine import attempted")
        return types.ModuleType(name)

    monkeypatch.setattr(sc.importlib, "import_module", guarded_import)

    try:
        sc.run_startup_checks(pyproject_path=str(pyproj), skip_stripe_router=True)
    finally:
        monkeypatch.setattr(sc.importlib, "import_module", original_import)

    assert captured["modules"] == []


def test_skip_stripe_router_env_prevents_stripe_lookup(monkeypatch):
    monkeypatch.setenv("MENACE_SKIP_STRIPE_ROUTER", "1")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_dummy")
    monkeypatch.setenv("STRIPE_PUBLIC_KEY", "pk_live_dummy")
    monkeypatch.setenv("STRIPE_ALLOWED_SECRET_KEYS", "sk_live_dummy")
    calls = {"count": 0}

    def _record(*args, **kwargs):
        calls["count"] += 1
        return {"id": "acct_1H123456789ABCDEF"}

    fake_stripe = types.SimpleNamespace(
        StripeClient=lambda api_key: types.SimpleNamespace(
            Account=types.SimpleNamespace(retrieve=_record)
        ),
        Account=types.SimpleNamespace(retrieve=_record),
    )
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    monkeypatch.setitem(sys.modules, "scpkg.stripe", fake_stripe)
    sys.modules.pop("stripe_billing_router", None)
    sys.modules.pop("scpkg.stripe_billing_router", None)

    module = _load("stripe_billing_router")

    assert calls["count"] == 0
    assert module._skip_stripe_verification()
    assert module._allowed_secret_keys() == set()


def test_verify_stripe_router_checks(monkeypatch):
    class FakeRegistry:
        def __init__(self, *a, **k):
            self.graph = types.SimpleNamespace(nodes=["finance_router_bot"])

    monkeypatch.setitem(
        sys.modules,
        "scpkg.bot_registry",
        types.SimpleNamespace(BotRegistry=FakeRegistry),
    )

    mod = types.SimpleNamespace(
        BILLING_RULES={("stripe", "default", "finance", "finance_router_bot"): {}},
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    setattr(mod, "STRIPE_" "SECRET_KEY", "sk")
    setattr(mod, "STRIPE_" "PUBLIC_KEY", "pk")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)
    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: "")

    def _run_ok(cmd, **k):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", _run_ok)
    sc.verify_stripe_router()

    bad = types.SimpleNamespace(BILLING_RULES={})
    setattr(bad, "STRIPE_" "SECRET_KEY", "")
    setattr(bad, "STRIPE_" "PUBLIC_KEY", "")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", bad)
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router()


def test_verify_stripe_router_missing_route(monkeypatch):
    class FakeRegistry:
        def __init__(self, *a, **k):
            self.graph = types.SimpleNamespace(
                nodes=["finance_router_bot", "unrouted_bot"]
            )

    monkeypatch.setitem(
        sys.modules,
        "scpkg.bot_registry",
        types.SimpleNamespace(BotRegistry=FakeRegistry),
    )

    mod = types.SimpleNamespace(
        BILLING_RULES={("stripe", "default", "finance", "finance_router_bot"): {}},
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    setattr(mod, "STRIPE_" "SECRET_KEY", "sk")
    setattr(mod, "STRIPE_" "PUBLIC_KEY", "pk")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)
    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: "")

    def _run_ok(cmd, **k):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", _run_ok)
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router()


def test_verify_stripe_router_required_bots(monkeypatch):
    class FakeRegistry:
        def __init__(self, *a, **k):
            self.graph = types.SimpleNamespace(nodes=[])

    monkeypatch.setitem(
        sys.modules,
        "scpkg.bot_registry",
        types.SimpleNamespace(BotRegistry=FakeRegistry),
    )

    called: list[str] = []

    def fake_resolve(bot_id, overrides=None):
        called.append(bot_id)
        if bot_id == "finance:finance_router_bot":
            return {}
        raise RuntimeError("missing route")

    mod = types.SimpleNamespace(
        BILLING_RULES={("stripe", "default", "finance", "finance_router_bot"): {}},
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
        _resolve_route=fake_resolve,
    )
    setattr(mod, "STRIPE_" "SECRET_KEY", "sk")
    setattr(mod, "STRIPE_" "PUBLIC_KEY", "pk")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)
    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: "")

    def _run_ok(cmd, **k):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", _run_ok)

    sc.verify_stripe_router(["finance:finance_router_bot"])
    assert called == ["finance:finance_router_bot"]
    called.clear()
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router([
            "finance:finance_router_bot",
            "finance:missing_bot",
        ])
    assert called == ["finance:finance_router_bot", "finance:missing_bot"]


def test_verify_stripe_router_import_scan(monkeypatch):
    class FakeRegistry:
        def __init__(self, *a, **k):
            self.graph = types.SimpleNamespace(nodes=["finance_router_bot"])

    monkeypatch.setitem(
        sys.modules,
        "scpkg.bot_registry",
        types.SimpleNamespace(BotRegistry=FakeRegistry),
    )

    mod = types.SimpleNamespace(
        BILLING_RULES={("stripe", "default", "finance", "finance_router_bot"): {}},
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    setattr(mod, "STRIPE_" "SECRET_KEY", "sk")
    setattr(mod, "STRIPE_" "PUBLIC_KEY", "pk")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)

    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: "")

    def _run(cmd, **k):
        if "check_stripe_imports.py" in cmd[1]:  # path-ignore
            return types.SimpleNamespace(returncode=1, stdout="bad", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", _run)
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router()


def test_verify_stripe_router_raw_usage_scan(monkeypatch):
    class FakeRegistry:
        def __init__(self, *a, **k):
            self.graph = types.SimpleNamespace(nodes=["finance_router_bot"])

    monkeypatch.setitem(
        sys.modules,
        "scpkg.bot_registry",
        types.SimpleNamespace(BotRegistry=FakeRegistry),
    )

    mod = types.SimpleNamespace(
        BILLING_RULES={("stripe", "default", "finance", "finance_router_bot"): {}},
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    setattr(mod, "STRIPE_" "SECRET_KEY", "sk")
    setattr(mod, "STRIPE_" "PUBLIC_KEY", "pk")
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)

    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: "")

    def _run(cmd, **k):
        if "check_raw_stripe_usage.py" in cmd[1]:  # path-ignore
            return types.SimpleNamespace(returncode=1, stdout="bad", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", _run)
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router()


def test_verify_optional_dependencies_reports_missing(monkeypatch):
    def _raise(*a, **k):
        raise ImportError

    monkeypatch.setattr(sc.importlib, "import_module", _raise)
    missing = sc.verify_optional_dependencies(["foo", "bar"])
    assert missing == ["foo", "bar"]
