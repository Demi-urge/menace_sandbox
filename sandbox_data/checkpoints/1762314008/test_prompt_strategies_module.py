import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

from menace_sandbox.self_improvement.prompt_strategies import (  # noqa: E402
    PromptStrategy,
    render_prompt,
)


def test_render_prompt_includes_module_name():
    ctx = {"module": "demo.py"}  # path-ignore
    out = render_prompt(PromptStrategy.STRICT_FIX, ctx)
    assert "demo.py" in out  # path-ignore
