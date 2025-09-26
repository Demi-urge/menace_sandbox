import menace.config as config


def _reset_config(monkeypatch):
    monkeypatch.setattr(config, "CONFIG", None)
    monkeypatch.setattr(config, "_CONFIG_STORE", None)
    monkeypatch.setattr(config, "UnifiedConfigStore", None, raising=False)


def test_stack_env_overrides_enable_streaming(monkeypatch):
    monkeypatch.setenv("MENACE_MODE", "dev")
    monkeypatch.setenv("STACK_STREAMING", "1")
    monkeypatch.setenv("STACK_INDEX_PATH", "/tmp/stack.index")
    monkeypatch.setenv("STACK_METADATA_PATH", "/tmp/stack.db")
    _reset_config(monkeypatch)

    cfg = config.load_config()

    assert cfg.stack_dataset.enabled is True
    assert cfg.context_builder.stack.enabled is True
    assert cfg.stack_dataset.index_path == "/tmp/stack.index"
    assert cfg.stack_dataset.metadata_path == "/tmp/stack.db"
    assert cfg.context_builder.stack.index_path == "/tmp/stack.index"
    assert cfg.context_builder.stack.metadata_path == "/tmp/stack.db"


def test_stack_env_defaults_when_unset(monkeypatch):
    monkeypatch.setenv("MENACE_MODE", "dev")
    for name in [
        "STACK_STREAMING",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
        "STACK_CACHE_DIR",
        "STACK_PROGRESS_PATH",
    ]:
        monkeypatch.delenv(name, raising=False)
    _reset_config(monkeypatch)

    cfg = config.load_config()

    assert cfg.stack_dataset.enabled is False
    assert cfg.context_builder.stack.enabled is False
    assert cfg.stack_dataset.index_path
    assert cfg.stack_dataset.metadata_path
