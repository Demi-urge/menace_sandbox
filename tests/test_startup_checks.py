import sys
import logging
import types
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"scpkg.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"scpkg.{name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pkg = types.ModuleType("scpkg")
pkg.__path__ = [str(ROOT)]
import importlib.machinery
pkg.__spec__ = importlib.machinery.ModuleSpec("scpkg", loader=None, is_package=True)
sys.modules["scpkg"] = pkg

AuditTrail = _load("audit_trail").AuditTrail
_load("dependency_verifier")
sc = _load("startup_checks")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import base64


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
    sc.run_startup_checks(pyproject_path=pyproj)
    assert "Missing required dependencies" in caplog.text


def test_optional_dependency_install(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "test")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, [])

    called: list[str] = []

    monkeypatch.setattr(sc, "validate_dependencies", lambda modules=sc.OPTIONAL_LIBS: ["missing_pkg"])
    monkeypatch.setattr(sc, "verify_project_dependencies", lambda p: [])
    monkeypatch.setattr(sc, "_install_packages", lambda pkgs: called.extend(pkgs))
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)

    sc.run_startup_checks(pyproject_path=pyproj)

    assert "missing_pkg" in called


def test_run_startup_checks_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "production")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["fake_package_456"])
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        sc.run_startup_checks(pyproject_path=pyproj)


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
    sc.run_startup_checks(pyproject_path=pyproj)
    # Corrupt the log
    with open(path, "a") as fh:
        fh.write("bad entry\n")
    monkeypatch.setattr(sc, "verify_optional_dependencies", lambda: [])
    monkeypatch.setattr(sc, "verify_stripe_router", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        sc.run_startup_checks(pyproject_path=pyproj)


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

    sc.run_startup_checks(pyproject_path=pyproj)

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

    sc.run_startup_checks(pyproject_path=pyproj)

    assert called["val"]


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
        STRIPE_SECRET_KEY="sk",
        STRIPE_PUBLIC_KEY="pk",
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)
    sc.verify_stripe_router()

    bad = types.SimpleNamespace(BILLING_RULES={}, STRIPE_SECRET_KEY="", STRIPE_PUBLIC_KEY="")
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
        STRIPE_SECRET_KEY="sk",
        STRIPE_PUBLIC_KEY="pk",
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
    )
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)
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
        STRIPE_SECRET_KEY="sk",
        STRIPE_PUBLIC_KEY="pk",
        ROUTING_TABLE={("stripe", "default", "finance", "finance_router_bot"): {}},
        _resolve_route=fake_resolve,
    )
    monkeypatch.setitem(sys.modules, "scpkg.stripe_billing_router", mod)

    sc.verify_stripe_router(["finance:finance_router_bot"])
    assert called == ["finance:finance_router_bot"]
    called.clear()
    with pytest.raises(RuntimeError):
        sc.verify_stripe_router([
            "finance:finance_router_bot",
            "finance:missing_bot",
        ])
    assert called == ["finance:finance_router_bot", "finance:missing_bot"]


def test_verify_optional_dependencies_reports_missing(monkeypatch):
    def _raise(*a, **k):
        raise ImportError

    monkeypatch.setattr(sc.importlib, "import_module", _raise)
    missing = sc.verify_optional_dependencies(["foo", "bar"])
    assert missing == ["foo", "bar"]

