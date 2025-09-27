import importlib

import importlib

import pytest

import config as config_module


@pytest.fixture(autouse=True)
def _reload_config_module():
    # Ensure each test gets a fresh view of environment driven settings.
    importlib.reload(config_module)
    yield
    importlib.reload(config_module)


def test_stack_dataset_overrides(monkeypatch):
    overrides = {
        "context_builder": {
            "stack_dataset": {
                "enabled": True,
                "dataset_name": "bigcode/test",
                "split": "eval",
                "ingestion": {
                    "languages": ["python", "rust"],
                    "max_document_lines": 512,
                    "chunk_overlap": 32,
                    "streaming": True,
                },
                "retrieval": {
                    "top_k": 12,
                    "weight": 2.5,
                    "max_context_documents": 6,
                    "max_context_lines": 900,
                },
                "cache": {
                    "data_dir": "/tmp/stack",
                    "index_path": "/tmp/stack/index.faiss",
                    "metadata_path": "/tmp/stack/meta.db",
                    "document_cache": "/tmp/stack/docs.cache",
                },
                "tokens": {
                    "env_vars": ["HF_TOKEN", "STACK_HF_TOKEN"],
                    "required": True,
                },
            }
        }
    }

    cfg = config_module.load_config(mode="dev", overrides=overrides)
    stack = cfg.context_builder.stack_dataset
    assert stack is not None
    assert stack.enabled is True
    assert stack.dataset_name == "bigcode/test"
    assert stack.split == "eval"
    assert stack.ingestion.languages == ["python", "rust"]
    assert stack.ingestion.max_document_lines == 512
    assert stack.ingestion.chunk_overlap == 32
    assert stack.ingestion.streaming is True
    assert stack.retrieval.top_k == 12
    assert pytest.approx(stack.retrieval.weight, rel=1e-9) == 2.5
    assert stack.retrieval.max_context_documents == 6
    assert stack.retrieval.max_context_lines == 900
    assert stack.cache.data_dir == "/tmp/stack"
    assert stack.cache.index_path == "/tmp/stack/index.faiss"
    assert stack.cache.metadata_path == "/tmp/stack/meta.db"
    assert stack.cache.document_cache == "/tmp/stack/docs.cache"
    assert stack.tokens.env_vars == ["HF_TOKEN", "STACK_HF_TOKEN"]
    assert stack.tokens.required is True

    stack_cfg = cfg.context_builder.stack
    assert stack_cfg is not None
    assert stack_cfg.enabled is True
    assert stack_cfg.dataset_name == "bigcode/test"
    assert stack_cfg.split == "eval"
    assert stack_cfg.languages == ["python", "rust"]
    assert stack_cfg.max_lines == 512
    assert stack_cfg.chunk_overlap == 32
    assert stack_cfg.streaming is True
    assert stack_cfg.top_k == 12
    assert pytest.approx(stack_cfg.weight, rel=1e-9) == 2.5
    assert stack_cfg.cache_dir == "/tmp/stack"
    assert stack_cfg.index_path == "/tmp/stack/index.faiss"
    assert stack_cfg.metadata_path == "/tmp/stack/meta.db"


def test_stack_dataset_environment(monkeypatch):
    monkeypatch.setenv("STACK_DATA_ENABLED", "1")
    monkeypatch.setenv("STACK_DATASET", "bigcode/the-stack-test")
    monkeypatch.setenv("STACK_SPLIT", "validation")
    monkeypatch.setenv("STACK_LANGUAGES", "Python,Go")
    monkeypatch.setenv("STACK_STREAMING", "true")
    monkeypatch.setenv("STACK_MAX_LINES", "256")
    monkeypatch.setenv("STACK_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("STACK_TOP_K", "33")
    monkeypatch.setenv("STACK_WEIGHT", "0.75")
    monkeypatch.setenv("STACK_CONTEXT_DOCS", "5")
    monkeypatch.setenv("STACK_CONTEXT_LINES", "640")
    monkeypatch.setenv("STACK_DATA_DIR", "/var/cache/stack")
    monkeypatch.setenv("STACK_CACHE_DIR", "/var/cache/stack")
    monkeypatch.setenv("STACK_VECTOR_PATH", "/var/cache/stack/index")
    monkeypatch.setenv("STACK_METADATA_DB", "/var/cache/stack/meta.sqlite")
    monkeypatch.setenv("STACK_METADATA_PATH", "/var/cache/stack/meta.sqlite")
    monkeypatch.setenv("STACK_DOCUMENT_CACHE", "/var/cache/stack/docs")

    cfg = config_module.load_config(mode="dev")
    stack = cfg.context_builder.stack_dataset
    assert stack is not None
    assert stack.enabled is True
    assert stack.dataset_name == "bigcode/the-stack-test"
    assert stack.split == "validation"
    assert stack.ingestion.languages == ["python", "go"]
    assert stack.ingestion.streaming is True
    assert stack.ingestion.max_document_lines == 256
    assert stack.ingestion.chunk_overlap == 64
    assert stack.retrieval.top_k == 33
    assert pytest.approx(stack.retrieval.weight, rel=1e-9) == 0.75
    assert stack.retrieval.max_context_documents == 5
    assert stack.retrieval.max_context_lines == 640
    assert stack.cache.data_dir == "/var/cache/stack"
    assert stack.cache.index_path == "/var/cache/stack/index"
    assert stack.cache.metadata_path == "/var/cache/stack/meta.sqlite"
    assert stack.cache.document_cache == "/var/cache/stack/docs"
    # Tokens fall back to defaults when not provided via environment overrides.
    assert "HF_TOKEN" in stack.tokens.env_vars

    stack_cfg = cfg.context_builder.stack
    assert stack_cfg is not None
    assert stack_cfg.enabled is True
    assert stack_cfg.dataset_name == "bigcode/the-stack-test"
    assert stack_cfg.split == "validation"
    assert stack_cfg.languages == ["python", "go"]
    assert stack_cfg.streaming is True
    assert stack_cfg.max_lines == 256
    assert stack_cfg.chunk_overlap == 64
    assert stack_cfg.top_k == 33
    assert pytest.approx(stack_cfg.weight, rel=1e-9) == 0.75
    assert stack_cfg.cache_dir == "/var/cache/stack"
    assert stack_cfg.index_path == "/var/cache/stack/index"
    assert stack_cfg.metadata_path == "/var/cache/stack/meta.sqlite"
