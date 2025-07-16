import logging
import os
from pathlib import Path
from menace.auto_env_setup import ensure_env, DEFAULT_VARS, interactive_setup
from menace.secrets_manager import SecretsManager
import menace.config_discovery as cd


def test_ensure_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    ensure_env(str(env))
    assert env.exists()
    data = env.read_text()
    assert "DATABASE_URL=" in data
    assert os.environ.get("MENACE_ENV_FILE") == str(env)


def test_generate_defaults(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    ensure_env(str(env))
    data = env.read_text()
    for var in cd._DEFAULT_VARS + list(DEFAULT_VARS):
        assert f"{var}=" in data
        assert os.environ.get(var)


def test_vault_usage(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)

    class DummyVault:
        def __init__(self):
            self.calls = []

        def export_env(self, name):
            self.calls.append(name)
            os.environ[name] = f"val-{name}"

    monkeypatch.setenv("SECRET_VAULT_URL", "http://vault")
    monkeypatch.setattr(
        "menace.auto_env_setup.VaultSecretProvider", lambda: DummyVault()
    )
    ensure_env(str(env))
    text = env.read_text()
    for var in cd._DEFAULT_VARS:
        assert f"{var}=val-{var}" in text
        assert os.environ.get(var) == f"val-{var}"


def test_vault_failure_logged(tmp_path, monkeypatch, caplog):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)

    class DummyVault:
        def export_env(self, name):
            raise RuntimeError("fail")

    monkeypatch.setenv("SECRET_VAULT_URL", "http://vault")
    monkeypatch.setattr(
        "menace.auto_env_setup.VaultSecretProvider", lambda: DummyVault()
    )
    caplog.set_level(logging.ERROR)
    ensure_env(str(env))
    for var in cd._DEFAULT_VARS:
        assert os.environ.get(var)
    assert "failed exporting" in caplog.text

def test_interactive_setup_generates(tmp_path):
    mgr = SecretsManager(str(tmp_path / "secrets.json"))
    os.environ.pop("A", None)
    os.environ.pop("B", None)
    interactive_setup(["A", "B"], secrets=mgr)
    assert "a" in mgr.secrets
    assert os.environ["A"] == mgr.secrets["a"]


def test_interactive_setup_defaults(tmp_path):
    mgr = SecretsManager(str(tmp_path / "secrets.json"))
    defaults = tmp_path / "defaults.env"
    defaults.write_text("B=foo\n")
    os.environ.pop("A", None)
    os.environ.pop("B", None)
    interactive_setup(["A", "B"], secrets=mgr, defaults_file=str(defaults))
    assert os.environ["B"] == "foo"
    assert mgr.secrets["b"] == "foo"

