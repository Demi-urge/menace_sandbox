import logging
import os
from pathlib import Path
from menace.auto_env_setup import ensure_env, DEFAULT_VARS, interactive_setup
from menace.secrets_manager import SecretsManager
import menace.config_discovery as cd
import json


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
    assert "SELF_TEST_RECURSIVE_ORPHANS=1" in data
    assert os.environ.get("SELF_TEST_RECURSIVE_ORPHANS") == "1"
    assert "SELF_TEST_RECURSIVE_ISOLATED=1" in data
    assert os.environ.get("SELF_TEST_RECURSIVE_ISOLATED") == "1"


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


def test_defaults_file_merge(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    defaults = tmp_path / "def.env"
    defaults.write_text("FOO=bar\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_DEFAULTS_FILE", str(defaults))
    ensure_env(str(env))
    text = env.read_text()
    assert "FOO=bar" in text
    assert os.environ["FOO"] == "bar"


def test_history_and_presets(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "roi_history.json").write_text("[0.1, 0.2]")
    (data_dir / "presets.json").write_text('[{"CPU_LIMIT":"2"}]')
    monkeypatch.chdir(tmp_path)
    ensure_env(str(env))
    text = env.read_text()
    assert "ROI_THRESHOLD=0.2" in text
    presets = json.loads(os.environ["SANDBOX_ENV_PRESETS"])
    assert presets[0]["CPU_LIMIT"] == "2"

