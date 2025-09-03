"""Run the personal sandbox with an optional local visual agent."""
from __future__ import annotations

import os
import time

from dynamic_path_router import resolve_path, path_for_prompt

from visual_agent_manager import VisualAgentManager
import run_autonomous


_DEF_PORT = "8001"


def _start_agent(manager: VisualAgentManager) -> bool:
    """Start the visual agent if it's not already running."""
    urls = os.environ.get("VISUAL_AGENT_URLS", f"http://127.0.0.1:{_DEF_PORT}")
    if run_autonomous._visual_agent_running(urls):
        return False
    token = os.environ.get("VISUAL_AGENT_TOKEN", "")
    manager.start(token)
    # Give the server a moment to come up
    for _ in range(10):
        if run_autonomous._visual_agent_running(urls):
            break
        time.sleep(0.5)
    return True


def _ensure_env() -> None:
    try:
        from auto_env_setup import ensure_env
    except Exception:  # pragma: no cover - import failure
        import auto_env_setup

        ensure_env = auto_env_setup.ensure_env  # type: ignore
    env_file = os.getenv("MENACE_ENV_FILE", ".env")
    ensure_env(env_file)


def main(argv: list[str] | None = None) -> None:
    """Start the visual agent if needed then run ``run_autonomous``."""
    _ensure_env()
    os.environ.setdefault("MENACE_AGENT_PORT", _DEF_PORT)
    os.environ.setdefault("VISUAL_AGENT_URLS", f"http://127.0.0.1:{_DEF_PORT}")

    manager = VisualAgentManager(path_for_prompt("menace_visual_agent_2.py"))
    started = _start_agent(manager)
    try:
        run_autonomous.main(argv)
    finally:
        if started:
            manager.shutdown()


if __name__ == "__main__":  # pragma: no cover - convenience
    main()
