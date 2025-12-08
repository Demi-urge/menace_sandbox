"""Backward-compatible nested package shim.

Some launchers still reference ``menace_sandbox.menace_sandbox`` even though the
package now lives at the repository root. This shim preserves those import paths
by delegating to the top-level package.
"""

from pathlib import Path

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in __path__:
    __path__.append(str(_PARENT))
