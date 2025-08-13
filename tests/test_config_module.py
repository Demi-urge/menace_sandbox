import os
from types import SimpleNamespace

import pytest

import config


class DummyBus:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def publish(self, name: str, payload: object) -> None:
        self.events.append((name, payload))


def test_yaml_and_json_parsing(tmp_path):
    json_file = tmp_path / "override.json"
    json_file.write_text('{"logging": {"verbosity": "ERROR"}}')
    yaml_file = tmp_path / "override.yaml"
    yaml_file.write_text("bot:\n  epsilon: 0.33\n")

    cfg_json = config.load_config(mode="dev", config_file=json_file)
    assert cfg_json.logging.verbosity == "ERROR"

    cfg_yaml = config.load_config(mode="dev", config_file=yaml_file)
    assert cfg_yaml.bot.epsilon == 0.33


def test_mode_overlays():
    dev_cfg = config.load_config(mode="dev")
    prod_cfg = config.load_config(mode="prod")
    assert dev_cfg.paths.data_dir == "./data"
    assert dev_cfg.logging.verbosity == "DEBUG"
    assert prod_cfg.paths.data_dir == "/srv/igi/data"
    assert prod_cfg.logging.verbosity == "WARNING"


def test_cli_overrides():
    args = config.parse_args(
        [
            "--mode",
            "dev",
            "--config-override",
            "logging.verbosity=ERROR",
            "--config-override",
            "bot.epsilon=0.4",
        ]
    )
    overrides = config._build_overrides(args.config_override or [])
    cfg = config.load_config(mode=args.mode, overrides=overrides)
    assert cfg.logging.verbosity == "ERROR"
    assert cfg.bot.epsilon == 0.4


def test_secret_loading_env(monkeypatch):
    monkeypatch.setenv("MYSECRET", "abc")
    assert config.Config.get_secret("mysecret") == "abc"


def test_secret_loading_vault(monkeypatch):
    monkeypatch.delenv("MYSECRET", raising=False)

    class DummyVault:
        def get(self, name):
            return "vault-secret" if name == "MYSECRET" else None

    class DummyStore:
        vault = DummyVault()

    monkeypatch.setattr(config, "_CONFIG_STORE", DummyStore())
    assert config.Config.get_secret("mysecret") == "vault-secret"
    assert os.environ["MYSECRET"] == "vault-secret"


def test_hot_reload_events(monkeypatch):
    bus = DummyBus()
    monkeypatch.setattr(config, "_EVENT_BUS", bus)

    reloaded = {"called": False}

    def fake_reload():
        reloaded["called"] = True
        bus.publish("config.reload", {})

    monkeypatch.setattr(config, "reload", fake_reload)
    handler = config._ConfigChangeHandler()
    event = SimpleNamespace(is_directory=False, src_path="cfg.yaml")
    handler.on_modified(event)

    assert reloaded["called"]
    assert ("config:file_change", {"path": "cfg.yaml"}) in bus.events
    assert any(name == "config.reload" for name, _ in bus.events)

