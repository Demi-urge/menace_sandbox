"""Run the personal sandbox."""
from __future__ import annotations

import os

import run_autonomous


def _ensure_env() -> None:
    try:
        from auto_env_setup import ensure_env
    except Exception:  # pragma: no cover - import failure
        import auto_env_setup

        ensure_env = auto_env_setup.ensure_env  # type: ignore
    env_file = os.getenv("MENACE_ENV_FILE", ".env")
    ensure_env(env_file)


def main(argv: list[str] | None = None) -> None:
    """Run ``run_autonomous`` after ensuring environment configuration."""
    _ensure_env()
    run_autonomous.main(argv)


if __name__ == "__main__":  # pragma: no cover - convenience
    main()
