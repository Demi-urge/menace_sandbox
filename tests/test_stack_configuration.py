from menace import config


def _reset_config():
    config.CONFIG = None
    config._OVERRIDES = {}
    config._MODE = None
    config._CONFIG_PATH = None


def test_context_builder_stack_defaults(monkeypatch):
    _reset_config()
    cfg = config.load_config(mode="dev")
    try:
        stack = cfg.context_builder.stack
        assert stack.enabled is False
        assert stack.languages == {"python", "javascript"}
        assert stack.max_lines == 200
        assert stack.max_bytes == 65536
        assert stack.retrieval_top_k == 3
        assert stack.index_path == "stack/index.faiss"
        assert stack.metadata_path == "stack/metadata.db"
        assert stack.cache_dir == "stack/cache"
        assert stack.progress_path == "stack/cache/progress.sqlite"
        assert stack.chunk_lines == 256
        assert cfg.context_builder.stack_prompt_enabled is False
        assert cfg.context_builder.stack_prompt_limit == 2
    finally:
        _reset_config()


def test_context_builder_stack_roundtrip(monkeypatch):
    _reset_config()
    cfg = config.load_config(mode="dev")
    overrides = {
        "context_builder": {
            "stack": {
                "enabled": True,
                "languages": ["python", "rust"],
                "max_lines": 64,
                "max_bytes": 4096,
                "retrieval_top_k": 7,
                "index_path": "custom/index.faiss",
                "metadata_path": "custom/meta.db",
                "cache_dir": "custom/cache",
                "progress_path": "custom/cache/progress.sqlite",
                "chunk_lines": 128,
            },
            "stack_prompt_enabled": True,
            "stack_prompt_limit": 5,
        }
    }
    updated = cfg.apply_overrides(overrides)
    try:
        stack = updated.context_builder.stack
        assert stack.enabled is True
        assert stack.languages == {"python", "rust"}
        assert stack.max_lines == 64
        assert stack.max_bytes == 4096
        assert stack.retrieval_top_k == 7
        assert stack.index_path == "custom/index.faiss"
        assert stack.metadata_path == "custom/meta.db"
        assert stack.cache_dir == "custom/cache"
        assert stack.progress_path == "custom/cache/progress.sqlite"
        assert stack.chunk_lines == 128
        assert updated.context_builder.stack_prompt_enabled is True
        assert updated.context_builder.stack_prompt_limit == 5
    finally:
        _reset_config()
