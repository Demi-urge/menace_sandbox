import os
import sys
import types
import subprocess

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
primitives_mod = sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
asym_mod = sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
ed_mod = types.ModuleType("ed25519")


class _DummyKey:
    def __init__(self, *a, **k):
        pass

    @staticmethod
    def from_private_bytes(b):
        return _DummyKey()

    def public_key(self):
        return _DummyKey()

    def public_bytes(self, *a, **k):
        return b""

    def sign(self, m):
        return b""


ed_mod.Ed25519PrivateKey = _DummyKey
ed_mod.Ed25519PublicKey = _DummyKey
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", ed_mod
)

ser_mod = types.ModuleType("serialization")


class _Encoding:
    Raw = object()


class _PublicFormat:
    Raw = object()


ser_mod.Encoding = _Encoding
ser_mod.PublicFormat = _PublicFormat
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", ser_mod
)

primitives_mod.asymmetric = asym_mod
primitives_mod.serialization = ser_mod
asym_mod.ed25519 = ed_mod
sys.modules["cryptography.hazmat"].primitives = primitives_mod

import light_bootstrap as lb
import menace.environment_bootstrap as eb


def test_methods_reused(monkeypatch):
    calls = []

    def dummy_export(self):
        calls.append("export")

    def dummy_migrations(self):
        calls.append("migrate")

    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", dummy_export)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", dummy_migrations)

    boot = lb.EnvironmentBootstrapper()
    boot.export_secrets()
    boot.run_migrations()

    assert calls == ["export", "migrate"]


def test_required_commands_extended():
    boot = lb.EnvironmentBootstrapper()
    for cmd in ["ssh", "pip", "alembic"]:
        assert cmd in boot.required_commands


def test_check_os_packages_pacman(monkeypatch):
    calls = []

    def fake_which(cmd):
        return "/usr/bin/pacman" if cmd == "pacman" else None

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(lb.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    boot = lb.EnvironmentBootstrapper()
    boot.check_os_packages(["pkg"])

    assert calls == [["pacman", "-Qi", "pkg"]]


def test_check_os_packages_apk(monkeypatch):
    calls = []

    def fake_which(cmd):
        return "/sbin/apk" if cmd == "apk" else None

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(lb.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    boot = lb.EnvironmentBootstrapper()
    boot.check_os_packages(["pkg"])

    assert calls == [["apk", "info", "-e", "pkg"]]

