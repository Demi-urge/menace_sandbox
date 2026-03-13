from __future__ import annotations

import importlib
import types


def test_auto_configure_env_readonly_skips_env_file_mutation(monkeypatch, tmp_path):
    bootstrap = importlib.import_module("sandbox_runner.bootstrap")

    env_file = tmp_path / ".env.runtime"
    env_file.write_text("", encoding="utf-8")

    ensure_calls: list[str] = []
    default_calls: list[str] = []

    def _fake_ensure_env(path: str) -> None:
        ensure_calls.append(path)

    class _FakeDefaultConfigManager:
        def __init__(self, path: str) -> None:
            self.path = path

        def apply_defaults(self) -> None:
            default_calls.append(self.path)

    settings = types.SimpleNamespace(
        menace_env_file=str(env_file),
        required_env_vars=(),
        sandbox_data_dir=str(tmp_path / "data"),
    )

    monkeypatch.setenv("MENACE_ENV_READONLY", "1")
    monkeypatch.setenv("MODELS", str(tmp_path))
    monkeypatch.setattr(bootstrap, "ensure_env", _fake_ensure_env)
    monkeypatch.setattr(bootstrap, "DefaultConfigManager", _FakeDefaultConfigManager)

    bootstrap.auto_configure_env(settings)

    assert ensure_calls == []
    assert default_calls == []
    assert env_file.read_text(encoding="utf-8") == ""
