#!/usr/bin/env python3
"""Launch the local visual agent and run the autonomous sandbox."""
from __future__ import annotations

import os
import subprocess
import sys
import time

from dynamic_path_router import resolve_path, path_for_prompt

import auto_env_setup
import run_autonomous


_DEF_PORT = "8001"


def _start_agent(env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Spawn ``menace_visual_agent_2.py`` with ``env``."""
    cmd = [sys.executable, path_for_prompt("menace_visual_agent_2.py")]
    if env.get("VISUAL_AGENT_AUTO_RECOVER") == "1":
        cmd.append("--auto-recover")
    return subprocess.Popen(cmd, env=env)


def _stop_agent(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> None:
    """Ensure environment, start the agent and run ``run_autonomous``."""
    env_file = os.getenv("MENACE_ENV_FILE", ".env")
    auto_env_setup.ensure_env(env_file)

    port = os.environ.get("MENACE_AGENT_PORT", _DEF_PORT)
    os.environ.setdefault("MENACE_AGENT_PORT", port)
    os.environ.setdefault("VISUAL_AGENT_URLS", f"http://127.0.0.1:{port}")

    agent_env = os.environ.copy()
    proc = _start_agent(agent_env)
    time.sleep(2)
    try:
        run_autonomous.main(argv)
    finally:
        _stop_agent(proc)


if __name__ == "__main__":
    main()
