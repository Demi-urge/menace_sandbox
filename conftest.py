import sys
import types
from pathlib import Path
import os

sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.__path__ = [str(Path(__file__).resolve().parent / "sandbox_runner")]
sys.modules.setdefault("sandbox_runner", sandbox_runner_pkg)

os.environ.setdefault("VISUAL_AGENT_URLS", "http://127.0.0.1:8001")
os.environ.setdefault("VISUAL_AGENT_TOKEN", "test-token")
