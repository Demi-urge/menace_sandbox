"""Backward-compatible nested package shim.

Some launchers still reference ``menace_sandbox.menace_sandbox`` even though the
package now lives at the repository root. This shim preserves those import paths
by delegating to the top-level package.
"""

# Intentionally empty; imports are handled by submodules.
