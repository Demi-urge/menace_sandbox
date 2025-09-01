import sys
import types
from pathlib import Path

base = Path(__file__).resolve().parent
pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(base / "sandbox_runner")]
root = types.ModuleType("menace_sandbox")
root.__path__ = [str(base)]
sys.modules.setdefault("sandbox_runner", pkg)
sys.modules.setdefault("menace_sandbox", root)
sys.modules.setdefault("menace_sandbox.sandbox_runner", pkg)
