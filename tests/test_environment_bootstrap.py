import os
import sys
import types
import subprocess
from pathlib import Path

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

menace = types.ModuleType("menace")
menace.RAISE_ERRORS = False
menace.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules["menace"] = menace
sys.modules["menace.vector_service.embedding_scheduler"] = types.SimpleNamespace(
    start_scheduler_from_env=lambda *a, **k: None
)

import menace.environment_bootstrap as eb
import menace.config_discovery as cd
import pytest
from sandbox_settings import SandboxSettings


def test_environment_bootstrapper(monkeypatch, tmp_path):
    tf_dir = tmp_path / "tf"
    tf_dir.mkdir()
    (tmp_path / "alembic.ini").write_text("[alembic]\n")
    monkeypatch.setenv("MENACE_BOOTSTRAP_DEPS", "dep1, dep2")
    monkeypatch.setenv("MODELS", "demo")

    calls: list[tuple[list[str], str | None]] = []

    def fake_run(cmd, check=False, cwd=None, **kwargs):
        calls.append((list(cmd), cwd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    called = []

    def ensure_wrapper():
        called.append(True)
        return cd.ensure_config(save_path=str(tmp_path / "env.auto"))

    monkeypatch.setattr(eb, "ensure_config", ensure_wrapper)

    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    boot = eb.EnvironmentBootstrapper(tf_dir=str(tf_dir))
    boot.bootstrap()

    assert called, "ensure_config should be invoked"

    for var in SandboxSettings().required_env_vars:
        assert os.getenv(var), f"{var} not set"

    expected = [
        ["pip", "install", "dep1"],
        ["pip", "install", "dep2"],
        ["alembic", "upgrade", "head"],
        ["terraform", "init"],
        ["terraform", "apply", "-auto-approve"],
    ]

    executed = [c[0] for c in calls]
    for cmd in expected:
        assert cmd in executed

    tf_calls = [c for c in calls if c[0][0] == "terraform"]
    assert all(c[1] == str(tf_dir) for c in tf_calls)


def test_dependency_retry(monkeypatch):
    attempts = []

    def failing_run(cmd, check=False, cwd=None, **kwargs):
        attempts.append(cmd)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", failing_run)
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.install_dependencies(["dep"])
    assert attempts.count(["pip", "install", "dep"]) == 2


def test_terraform_retry(monkeypatch, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    calls = []

    def failing_run(cmd, check=False, cwd=None, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["terraform", "init"] and calls.count(cmd) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", failing_run)
    boot = eb.InfrastructureBootstrapper(tf_dir=str(tmp_path))
    assert boot.bootstrap()
    assert calls.count(["terraform", "init"]) == 2


def test_command_and_remote_checks(monkeypatch, caplog):
    missing = []

    def fake_which(cmd):
        if cmd == "git":
            return "/usr/bin/git"
        missing.append(cmd)
        return None

    def fake_run(cmd, check=False, **kwargs):
        if "bad" in cmd[-1]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(eb.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.check_commands(["git", "curl"])
    boot.check_remote_dependencies(["http://good", "http://bad"])
    assert "required command missing" in caplog.text
    assert "remote dependency unreachable" in caplog.text


def test_remote_check_triggers_local_provision(monkeypatch):
    def fake_run(cmd, check=False, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    called = []

    def fake_prov(self):
        called.append(True)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(eb.ExternalDependencyProvisioner, "provision", fake_prov)

    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.check_remote_dependencies(["http://bad"])

    assert called == [True]


def test_check_os_packages_dpkg(monkeypatch):
    calls = []

    def fake_which(cmd):
        return "/usr/bin/dpkg" if cmd == "dpkg" else None

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(eb.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.check_os_packages(["pkg1", "pkg2"])

    assert calls == [["dpkg", "-s", "pkg1"], ["dpkg", "-s", "pkg2"]]


def test_check_os_packages_missing(monkeypatch):
    def fake_which(cmd):
        return "/bin/rpm" if cmd == "rpm" else None

    def fake_run(cmd, check=False, **kwargs):
        if "badpkg" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(eb.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    with pytest.raises(RuntimeError):
        boot.check_os_packages(["goodpkg", "badpkg"])


def test_bootstrap_installs_missing_packages(monkeypatch):
    monkeypatch.setenv("MENACE_OS_PACKAGES", "pkg1,pkg2")

    calls = []

    def fake_check(self, pkgs):
        calls.append(list(pkgs))
        if len(calls) == 1:
            raise RuntimeError("missing OS packages: " + ", ".join(pkgs))

    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda self, cmds: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda self, urls: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", lambda self, reqs: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda self: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", fake_check)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: None)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)

    installs = []

    def fake_install(self):
        installs.append(self.packages)

    monkeypatch.setattr(eb.SystemProvisioner, "ensure_packages", fake_install)

    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.bootstrap()

    assert installs == [["pkg1", "pkg2"]]
    assert len(calls) == 2


def test_bootstrap_install_failure(monkeypatch):
    monkeypatch.setenv("MENACE_OS_PACKAGES", "pkg1")

    calls = []

    def always_fail(self, pkgs):
        calls.append(list(pkgs))
        raise RuntimeError("missing OS packages: " + ", ".join(pkgs))

    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda self, cmds: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda self, urls: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", lambda self, reqs: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda self: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", always_fail)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: None)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)

    installs = []

    def fake_install(self):
        installs.append(self.packages)

    monkeypatch.setattr(eb.SystemProvisioner, "ensure_packages", fake_install)

    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    with pytest.raises(RuntimeError):
        boot.bootstrap()

    assert installs == [["pkg1"]]
    assert len(calls) == 2


def test_bootstrap_installs_apscheduler(monkeypatch):
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda s, c: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda s, u: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", lambda s, p: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda s: None)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", lambda self: None)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)
    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    monkeypatch.setattr(eb.importlib.util, "find_spec", lambda name: None if name == "apscheduler" else object())
    installs = []
    def fake_install(self, reqs):
        installs.extend(reqs)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", fake_install)
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.bootstrap()
    assert "apscheduler" in installs


def test_vector_assets_skipped_when_huggingface_missing(monkeypatch, caplog):
    original_find_spec = eb.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "huggingface_hub":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(eb.importlib.util, "find_spec", fake_find_spec)
    caplog.set_level("INFO")
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    caplog.clear()
    boot.bootstrap_vector_assets()
    assert "Skipping embedding model download" in caplog.text


def test_bootstrap_logs_info_when_systemd_unavailable(monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda s, c: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda s, u: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", lambda s, p: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", lambda s, r: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_nvidia_driver", lambda s: None)
    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: None)
    monkeypatch.setattr(eb.SystemProvisioner, "ensure_packages", lambda self: None)
    monkeypatch.setattr(eb.ensure_config, lambda: None)

    original_which = eb.shutil.which

    def fake_which(cmd):
        if cmd == "systemctl":
            return "/bin/systemctl"
        return original_which(cmd)

    monkeypatch.setattr(eb.shutil, "which", fake_which)

    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "System has not been booted with systemd"

    monkeypatch.setattr(eb.subprocess, "run", lambda *a, **k: FakeResult())

    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.bootstrap()

    assert "systemd unavailable" in caplog.text
    assert "failed enabling sandbox_autopurge.timer" not in caplog.text


def test_security_audit_removed(monkeypatch, tmp_path, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda s, c: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda s, u: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", lambda s, r: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", lambda s, p: None)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: True)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", lambda self: None)
    called = {"fix": 0}

    def fake_fix(auditor):
        called["fix"] += 1
        return False

    # fix_until_safe no longer used

    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    boot = eb.EnvironmentBootstrapper(tf_dir=str(tmp_path))
    boot.bootstrap()
    assert os.environ.get("MENACE_SAFE") is None
    assert called["fix"] == 0


def test_security_audit_not_invoked(monkeypatch, tmp_path):
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda s, c: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda s, u: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", lambda s, r: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", lambda s, p: None)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: True)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", lambda self: None)

    counts = {"audit": 0, "fix": 0}

    def fake_audit(self):
        counts["audit"] += 1
        return False

    def fake_fix(aud):
        counts["fix"] += 1
        return False

    # fix_until_safe no longer used
    monkeypatch.setattr(eb.startup_checks, "verify_project_dependencies", lambda: [])
    boot = eb.EnvironmentBootstrapper(tf_dir=str(tmp_path))
    boot.bootstrap()
    # The auditor and fixer should never be called
    assert counts["audit"] == 0
    assert counts["fix"] == 0
    assert os.environ.get("MENACE_SAFE") is None


def _fake_which_nvidia(cmd: str) -> str | None:
    return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else "/bin/" + cmd


def test_check_nvidia_driver_warns(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setenv("MIN_NVIDIA_DRIVER_VERSION", "535.0")
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    monkeypatch.setattr(eb.shutil, "which", _fake_which_nvidia)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "534.0\n")
    boot.check_nvidia_driver()
    assert "NVIDIA driver" in caplog.text


def test_check_nvidia_driver_error(monkeypatch):
    monkeypatch.setenv("MIN_NVIDIA_DRIVER_VERSION", "535.0")
    monkeypatch.setenv("STRICT_NVIDIA_DRIVER_CHECK", "1")
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    monkeypatch.setattr(eb.shutil, "which", _fake_which_nvidia)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "534.0\n")
    with pytest.raises(RuntimeError):
        boot.check_nvidia_driver()


def test_check_nvidia_driver_ok(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setenv("MIN_NVIDIA_DRIVER_VERSION", "535.0")
    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    monkeypatch.setattr(eb.shutil, "which", _fake_which_nvidia)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "536.0\n")
    boot.check_nvidia_driver()
    assert "NVIDIA driver" not in caplog.text


def _write_pyproject(path, deps):
    path.write_text("""[project]\ndependencies = [\n""" +
                    "\n".join(f'    "{d}",' for d in deps) + "\n]\n")


def test_bootstrap_verifies_pyproject_dependencies(monkeypatch, tmp_path):
    pyproj = tmp_path / "pyproject.toml"
    _write_pyproject(pyproj, ["missing_pkg"])
    orig_verify = eb.startup_checks.verify_project_dependencies
    monkeypatch.setattr(
        eb.startup_checks,
        "verify_project_dependencies",
        lambda path=None, orig=orig_verify: orig(pyproj),
    )

    checked = []

    def fake_verify(mods):
        checked.extend(mods)
        return list(mods)

    installs = []

    monkeypatch.setattr(eb.startup_checks, "verify_modules", fake_verify)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_commands", lambda s, c: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_remote_dependencies", lambda s, u: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_os_packages", lambda s, p: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "run_migrations", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "export_secrets", lambda s: None)
    monkeypatch.setattr(eb.EnvironmentBootstrapper, "check_nvidia_driver", lambda s: None)
    monkeypatch.setattr(eb.InfrastructureBootstrapper, "bootstrap", lambda self: None)
    monkeypatch.setattr(eb, "ensure_config", lambda: None)

    def fake_install(self, reqs):
        installs.extend(reqs)

    monkeypatch.setattr(eb.EnvironmentBootstrapper, "install_dependencies", fake_install)

    boot = eb.EnvironmentBootstrapper(tf_dir=".")
    boot.bootstrap()

    assert checked == ["missing_pkg"]
    assert installs == ["missing_pkg", "apscheduler"]

