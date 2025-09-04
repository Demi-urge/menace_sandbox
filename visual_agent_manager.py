from __future__ import annotations

from contextlib import suppress
import os
import signal
import subprocess
import sys
import tempfile
import time
import json
from pathlib import Path

from dynamic_path_router import resolve_path

try:
    from .visual_agent_queue import VisualAgentQueue
except ImportError:  # pragma: no cover - allow running as script
    from visual_agent_queue import VisualAgentQueue


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class VisualAgentManager:
    """Manage the lifecycle of ``menace_visual_agent_2.py``."""

    def __init__(self, agent_script: str | None = None) -> None:
        self.agent_script = agent_script or str(
            resolve_path("menace_visual_agent_2.py")
        )
        self.pid_file = Path(
            os.getenv(
                "VISUAL_AGENT_PID_FILE",
                os.path.join(tempfile.gettempdir(), "visual_agent.pid"),
            )
        )
        self.process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    def shutdown(self, timeout: float = 5.0) -> None:
        """Terminate the running visual agent if possible."""
        pid = None
        if self.process and self.process.poll() is None:
            pid = self.process.pid
        elif self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
            except Exception:
                pid = None
        if not pid:
            if self.pid_file.exists():
                with suppress(Exception):
                    self.pid_file.unlink()
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        start = time.time()
        while time.time() - start < timeout and _pid_alive(pid):
            time.sleep(0.1)
        if _pid_alive(pid):
            with suppress(Exception):
                os.kill(pid, signal.SIGKILL)
        with suppress(Exception):
            self.pid_file.unlink()
        if self.process:
            with suppress(Exception):
                self.process.wait(timeout=1)
            self.process = None

    # ------------------------------------------------------------------
    def start(self, token: str) -> subprocess.Popen:
        """Start the visual agent using ``token``."""
        env = os.environ.copy()
        env["VISUAL_AGENT_TOKEN"] = token
        data_dir = Path(resolve_path(env.get("SANDBOX_DATA_DIR", "sandbox_data")))
        queue_db = data_dir / "visual_agent_queue.db"
        queue = VisualAgentQueue(queue_db)
        recovered = False
        try:
            recovered = queue.check_integrity()
            queue.reset_running_tasks()
            if recovered:
                state_path = data_dir / "visual_agent_state.json"
                if state_path.exists():
                    try:
                        data = json.loads(state_path.read_text())
                        status = data.get("status", {})
                        if isinstance(status, dict):
                            for tid, info in status.items():
                                if (
                                    isinstance(info, dict)
                                    and info.get("status") in {"queued", "running"}
                                ):
                                    queue.append({
                                        "id": tid,
                                        "prompt": info.get("prompt", ""),
                                        "branch": info.get("branch"),
                                    })
                        lc = data.get("last_completed")
                        if isinstance(lc, (int, float)):
                            queue.set_last_completed(float(lc))
                    except Exception:
                        pass
        except Exception:
            pass

        cmd = [sys.executable, self.agent_script]
        auto = env.get("VISUAL_AGENT_AUTO_RECOVER", "1")
        if auto == "0":
            cmd.append("--no-auto-recover")
        elif auto == "1":
            cmd.append("--auto-recover")
        self.process = subprocess.Popen(cmd, env=env)
        return self.process

    # ------------------------------------------------------------------
    def restart_with_token(self, token: str) -> subprocess.Popen:
        """Restart the agent with ``token``."""
        self.shutdown()
        return self.start(token)

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """Return ``True`` if the visual agent process is alive."""
        pid: int | None = None
        if self.process and self.process.poll() is None:
            pid = self.process.pid
        elif self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
            except Exception:
                pid = None
        return bool(pid and _pid_alive(pid))
