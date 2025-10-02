from __future__ import annotations

from pathlib import Path

from bootstrap_defaults import ensure_bootstrap_defaults


def test_ensure_bootstrap_defaults_persists(tmp_path):
    required = [
        "DATABASE_URL",
        "OPENAI_API_KEY",
        "MENACE_EMAIL",
        "MENACE_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    env: dict[str, str] = {}

    created, env_file = ensure_bootstrap_defaults(
        required, repo_root=tmp_path, environ=env
    )
    assert created == set(required)
    assert env_file == Path(tmp_path) / ".env.bootstrap"
    for key in required:
        assert env[key]

    initial_values = {key: env[key] for key in required}

    created_again, env_file_again = ensure_bootstrap_defaults(
        required, repo_root=tmp_path, environ=env
    )
    assert not created_again
    assert env_file_again == env_file
    for key in required:
        assert env[key] == initial_values[key]

    content = env_file.read_text()
    for key in required:
        assert f"{key}=" in content
