from pathlib import Path

from sandbox_settings import SandboxSettings


def test_defaults(monkeypatch):
    monkeypatch.delenv("PROMPT_CHUNK_TOKEN_THRESHOLD", raising=False)
    monkeypatch.delenv("CHUNK_SUMMARY_CACHE_DIR", raising=False)
    monkeypatch.delenv("PROMPT_CHUNK_CACHE_DIR", raising=False)
    settings = SandboxSettings()
    assert settings.prompt_chunk_token_threshold == 3500
    assert settings.chunk_summary_cache_dir == Path("chunk_summary_cache")
    assert settings.prompt_chunk_cache_dir == Path("chunk_summary_cache")


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPT_CHUNK_TOKEN_THRESHOLD", "1234")
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("CHUNK_SUMMARY_CACHE_DIR", str(cache_dir))
    settings = SandboxSettings()
    assert settings.prompt_chunk_token_threshold == 1234
    assert settings.chunk_summary_cache_dir == cache_dir
    assert settings.prompt_chunk_cache_dir == cache_dir
