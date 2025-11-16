import os
from menace.default_config_manager import DefaultConfigManager
import menace.config_discovery as cd


def test_apply_defaults(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    mgr = DefaultConfigManager(str(env))
    mgr.apply_defaults()

    assert env.exists()
    data = env.read_text()
    for var in cd._DEFAULT_VARS:
        assert f"{var}=" in data
        assert os.environ.get(var)
    assert os.environ.get("MENACE_ENV_FILE") == str(env)


def test_preserve_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=abc\n")
    monkeypatch.chdir(tmp_path)
    for var in cd._DEFAULT_VARS:
        monkeypatch.delenv(var, raising=False)
    mgr = DefaultConfigManager(str(env))
    mgr.apply_defaults()

    data = env.read_text()
    assert "OPENAI_API_KEY=abc" in data
    assert os.environ.get("OPENAI_API_KEY") == "abc"
