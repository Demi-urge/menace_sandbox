import importlib
from pathlib import Path

import pytest


def test_database_url_normalizes_blank(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "   ")
    import env_config

    importlib.reload(env_config)

    assert env_config.DATABASE_URL == "sqlite:///menace.db"


def test_migration_env_uses_normalized_db_url():
    env_script = Path("migrations/env.py").read_text()
    assert "normalize_db_url" in env_script


def test_blank_database_url_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    import env_config

    importlib.reload(env_config)

    from databases import MenaceDB

    db = MenaceDB(url="")
    assert db.engine.url.get_backend_name() == "sqlite"


def test_invalid_database_url_falls_back_in_non_production(monkeypatch):
    monkeypatch.setenv("MENACE_MODE", "development")
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    import env_config

    importlib.reload(env_config)

    assert env_config.DATABASE_URL == "sqlite:///menace.db"


def test_bare_path_database_url_normalizes_to_sqlite(monkeypatch, tmp_path):
    db_path = tmp_path / "menace.db"
    monkeypatch.setenv("DATABASE_URL", str(db_path))
    import env_config

    importlib.reload(env_config)

    expected = f"sqlite:///{db_path.expanduser().resolve().as_posix()}"
    assert env_config.DATABASE_URL == expected


def test_invalid_database_url_raises_in_production(monkeypatch):
    monkeypatch.setenv("MENACE_MODE", "production")
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    import env_config

    with pytest.raises(ValueError):
        importlib.reload(env_config)
