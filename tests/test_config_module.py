import os
import json
import yaml
import pytest
from pydantic import ValidationError

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace import config


class DummyBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def publish(self, topic: str, payload: object) -> None:
        self.events.append((topic, payload))


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    settings = {
        "paths": {"data_dir": "/data", "log_dir": "/logs"},
        "thresholds": {"error": 0.1, "alert": 0.5},
        "api_keys": {"openai": "yaml_openai", "serp": "yaml_serp"},
        "logging": {"verbosity": "INFO"},
        "vector": {"dimensions": 8, "distance_metric": "cosine"},
        "bot": {"learning_rate": 0.01, "epsilon": 0.1},
        "watch_config": False,
    }
    profile = {"thresholds": {"alert": 0.7}, "logging": {"verbosity": "WARNING"}}
    extra = {"bot": {"learning_rate": 0.02}}
    (tmp_path / "settings.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    (tmp_path / "dev.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    (tmp_path / "extra.json").write_text(json.dumps(extra), encoding="utf-8")

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS_FILE", tmp_path / "settings.yaml")
    monkeypatch.setattr(config, "Observer", None)
    monkeypatch.setattr(config, "UnifiedConfigStore", None)
    monkeypatch.setattr(config, "CONFIG", None)
    monkeypatch.setattr(config, "_MODE", None)
    monkeypatch.setattr(config, "_CONFIG_PATH", None)
    monkeypatch.setattr(config, "_OVERRIDES", {})
    monkeypatch.setattr(config, "_EVENT_BUS", None)
    monkeypatch.setattr(config, "_WATCHER_ENABLED", False)
    return tmp_path


def test_yaml_json_parsing_and_schema_validation(config_env):
    cfg = config.load_config(mode="dev", config_file=config_env / "extra.json")
    assert cfg.logging.verbosity == "WARNING"
    assert cfg.bot.learning_rate == 0.02

    bad = config_env / "bad.json"
    bad.write_text(json.dumps({"bot": {"epsilon": 2.0}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        config.load_config(mode="dev", config_file=bad)


def test_mode_overlay_logic(config_env):
    cfg = config.load_config(mode="dev")
    assert cfg.thresholds.alert == 0.7
    assert cfg.logging.verbosity == "WARNING"


def test_cli_and_programmatic_overrides(config_env):
    cfg = config.load_config(mode="dev", overrides={"logging": {"verbosity": "DEBUG"}})
    assert cfg.logging.verbosity == "DEBUG"

    args = config.parse_args(["--mode", "dev", "logging.verbosity=ERROR"])
    overrides = config._build_overrides(args.overrides)
    cfg2 = config.load_config(mode=args.mode, overrides=overrides)
    assert cfg2.logging.verbosity == "ERROR"


def test_event_bus_broadcast_on_reload(config_env):
    bus = DummyBus()
    config.set_event_bus(bus)
    cfg = config.load_config(mode="dev", config_file=config_env / "extra.json")
    config.CONFIG = cfg
    config._MODE = "dev"
    config._CONFIG_PATH = config_env / "extra.json"
    dev_file = config_env / "dev.yaml"
    dev_file.write_text("thresholds:\n  alert: 0.8\n", encoding="utf-8")
    config.reload()
    assert bus.events
    topic, payload = bus.events[-1]
    assert topic == "config:reload"
    assert payload["diff"]["thresholds"]["alert"] == 0.8
    config.set_event_bus(None)


def test_secret_loading_precedence(config_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_openai")
    monkeypatch.setenv("SERP_API_KEY", "env_serp")

    class DummyVault:
        def get(self, key: str):
            return {"OPENAI_API_KEY": "vault_openai"}.get(key)

    monkeypatch.setattr(config, "VaultSecretProvider", DummyVault)
    cfg = config.load_config(mode="dev")
    assert cfg.api_keys.openai == "vault_openai"
    assert cfg.api_keys.serp == "env_serp"


def test_watchdog_triggers_reload(config_env, monkeypatch):
    events = DummyBus()
    monkeypatch.setattr(config, "_EVENT_BUS", events)
    called = []

    def fake_reload():
        called.append(True)

    monkeypatch.setattr(config, "reload", fake_reload)
    settings = str((config_env / "settings.yaml").resolve())
    handler = config._ConfigChangeHandler({settings})
    event = type("Evt", (), {"is_directory": False, "src_path": settings})
    handler.on_modified(event)
    assert called
    assert events.events
    assert events.events[-1][0] == "config:file_change"
    assert events.events[-1][1]["path"] == settings
