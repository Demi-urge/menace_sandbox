import os
import logging
import pytest

import menace.startup_checks as sc
from menace.audit_trail import AuditTrail
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

    sc.run_startup_checks(pyproject_path=pyproj)

    assert "missing_pkg" in called


def test_run_startup_checks_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_MODE", "production")
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["fake_package_456"])
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
    sc.run_startup_checks(pyproject_path=pyproj)
    # Corrupt the log
    with open(path, "a") as fh:
        fh.write("bad entry\n")
    with pytest.raises(RuntimeError):
        sc.run_startup_checks(pyproject_path=pyproj)

