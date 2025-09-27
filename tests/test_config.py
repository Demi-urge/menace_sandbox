import textwrap
from pathlib import Path

import pytest

import config as config_module
from config import Config
from pydantic import ValidationError


BASE_CONFIG = {
    "paths": {"data_dir": "/data", "log_dir": "/logs"},
    "thresholds": {"error": 0.1, "alert": 0.9},
    "api_keys": {"openai": "test-openai", "serp": "test-serp"},
    "logging": {"verbosity": "INFO"},
    "vector": {"dimensions": 256, "distance_metric": "cosine"},
    "vector_store": {"backend": "faiss", "path": "vectors.index"},
    "bot": {"learning_rate": 0.01, "epsilon": 0.1},
}


def test_context_builder_without_stack_section():
    config = Config.model_validate(BASE_CONFIG)
    assert config.context_builder.stack_dataset is None


def test_context_builder_with_stack_section():
    data = {
        **BASE_CONFIG,
        "context_builder": {
            "stack_dataset": {
                "enabled": True,
                "dataset_name": "bigcode/the-stack-v2-dedup",
                "languages": ["python", "rust"],
                "max_lines": 400,
                "chunk_overlap": 20,
                "top_k": 25,
                "weight": 1.5,
            }
        },
    }
    config = Config.model_validate(data)
    stack = config.context_builder.stack_dataset
    assert stack is not None
    assert stack.enabled is True
    assert stack.languages == ["python", "rust"]
    assert stack.top_k == 25
    assert stack.weight == pytest.approx(1.5)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "context_builder": {
                "stack_dataset": {
                    "languages": ["klingon"],
                }
            }
        },
        {
            "context_builder": {
                "stack_dataset": {
                    "max_lines": -5,
                }
            }
        },
    ],
)
def test_invalid_stack_dataset_values_raise(payload):
    data = {**BASE_CONFIG, **payload}
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_environment_overrides_merge_stack(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.yaml"
    settings_path.write_text(
        textwrap.dedent(
            """
            paths:
              data_dir: /tmp/data
              log_dir: /tmp/logs
            thresholds:
              error: 0.2
              alert: 0.8
            api_keys:
              openai: test-openai
              serp: test-serp
            logging:
              verbosity: INFO
            vector:
              dimensions: 128
              distance_metric: cosine
            vector_store:
              backend: faiss
              path: vectors.index
            bot:
              learning_rate: 0.05
              epsilon: 0.2
            """
        )
    )
    (config_dir / "dev.yaml").write_text("{}")

    monkeypatch.setenv("MENACE_MODE", "dev")
    monkeypatch.setenv("STACK_LANGUAGES", "python, rust")
    monkeypatch.setenv("STACK_TOP_K", "7")
    monkeypatch.setenv("STACK_DATA_ENABLED", "1")
    monkeypatch.setenv("STACK_DATA_INDEX", "/var/stack/index")
    monkeypatch.setenv("STACK_METADATA_DB", "/var/stack/meta.db")

    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "DEFAULT_SETTINGS_FILE", settings_path)
    monkeypatch.setattr(config_module, "CONFIG", None)
    monkeypatch.setattr(config_module, "_WATCHER", None)
    monkeypatch.setattr(config_module, "_CONFIG_PATH", None)
    monkeypatch.setattr(config_module, "_OVERRIDES", {})

    cfg = config_module.load_config()
    stack = cfg.context_builder.stack_dataset
    assert stack is not None
    assert stack.enabled is True
    assert stack.languages == ["python", "rust"]
    assert stack.top_k == 7
    assert stack.index_path == "/var/stack/index"
    assert stack.metadata_path == "/var/stack/meta.db"

    # Reset module state so subsequent tests are unaffected
    config_module.CONFIG = None
