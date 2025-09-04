import logging
import sqlite3
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402

from sandbox_settings import SandboxSettings  # noqa: E402

menace_pkg = types.ModuleType("menace")
auto_env = types.ModuleType("auto_env_setup")
auto_env.ensure_env = lambda *a, **k: None
default_cfg = types.ModuleType("default_config_manager")
default_cfg.DefaultConfigManager = object
menace_pkg.auto_env_setup = auto_env
menace_pkg.default_config_manager = default_cfg
sys.modules.setdefault("menace", menace_pkg)
sys.modules.setdefault("menace.auto_env_setup", auto_env)
sys.modules.setdefault("menace.default_config_manager", default_cfg)
cli_stub = types.ModuleType("sandbox_runner.cli")
cli_stub.main = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.cli", cli_stub)
cycle_stub = types.ModuleType("sandbox_runner.cycle")
cycle_stub.ensure_vector_service = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.cycle", cycle_stub)

from sandbox_runner import bootstrap  # noqa: E402


def _prepare(monkeypatch, tmp_path, db_names):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(repo))
    settings = SandboxSettings()
    settings.sandbox_repo_path = str(repo)
    settings.sandbox_data_dir = str(data)
    settings.sandbox_required_db_files = db_names
    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)
    return data


def test_sandbox_health_databases_ok(tmp_path, monkeypatch):
    data = _prepare(monkeypatch, tmp_path, ["ok.db"])
    path = data / "ok.db"
    sqlite3.connect(path).close()  # noqa: SQL001
    result = bootstrap.sandbox_health()
    assert result["databases_accessible"] is True
    assert result["database_errors"] == {}


def test_sandbox_health_databases_corrupted(tmp_path, monkeypatch, caplog):
    data = _prepare(monkeypatch, tmp_path, ["bad.db", "missing.db"])
    path = data / "bad.db"
    path.write_text("not a db", encoding="utf-8")
    # missing.db is intentionally absent
    with caplog.at_level(logging.ERROR):
        result = bootstrap.sandbox_health()
    assert result["databases_accessible"] is False
    assert set(result["database_errors"]) == {"bad.db", "missing.db"}
    assert "bad.db" in caplog.text and "missing.db" in caplog.text
