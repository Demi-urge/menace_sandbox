import sys
import types
from pathlib import Path

sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.__path__ = [str(Path(__file__).resolve().parent / "sandbox_runner")]
sys.modules.setdefault("sandbox_runner", sandbox_runner_pkg)
