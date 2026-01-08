import importlib
from pathlib import Path


def test_database_url_normalizes_blank(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "   ")
    import env_config

    importlib.reload(env_config)

    assert env_config.DATABASE_URL == "sqlite:///menace.db"


def test_migration_env_uses_normalized_db_url():
    env_script = Path("migrations/env.py").read_text()
    assert "env_config.DATABASE_URL" in env_script
