"""Backward-compatible nested package shim.

Some launchers still reference ``menace_sandbox.menace_sandbox`` even though the
package now lives at the repository root. This shim preserves those import paths
by delegating to the top-level package.
"""

from importlib import import_module
import importlib.util
from pathlib import Path
import sys

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in __path__:
    __path__.append(str(_PARENT))

# Expose ``context_builder_util`` as a top-level module while sourcing the
# implementation from the packaged location. This avoids accidental imports of
# similarly named files outside the package and keeps legacy import paths
# working.
try:  # pragma: no cover - defensive shim
    _ctx_builder_mod = import_module(f"{__name__}.context_builder_util")
except ModuleNotFoundError:  # pragma: no cover - defensive shim
    _ctx_builder_mod = None
else:  # pragma: no cover - lightweight module registration
    sys.modules.setdefault("context_builder_util", _ctx_builder_mod)

_self_improvement_spec = importlib.util.find_spec(f"{__name__}.self_improvement")
if _self_improvement_spec is not None:
    _self_improvement_mod = import_module(f"{__name__}.self_improvement")
    sys.modules.setdefault("self_improvement", _self_improvement_mod)
