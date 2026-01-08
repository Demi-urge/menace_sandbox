"""Backward-compatible nested package shim.

Some launchers still reference ``menace_sandbox.menace_sandbox`` even though the
package now lives at the repository root. This shim preserves those import paths
by delegating to the top-level package.
"""

from pathlib import Path

_PARENT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PARENT, _ROOT):
    if str(_path) not in __path__:
        __path__.append(str(_path))
