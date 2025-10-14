"""Shared provenance state to deduplicate unsigned provenance warnings.

This module centralises caches that were previously maintained separately
within modules such as :mod:`coding_bot_interface` and :mod:`bot_registry`.
When those modules are imported under multiple package aliases (for example
``menace`` and ``menace_sandbox``) the module-level globals are no longer
shared which resulted in repeated warnings and noisy log output.  By housing
these caches in a dedicated module we ensure that all imports reference the
same data structures regardless of the package prefix that was used.
"""

from __future__ import annotations

import threading

# Locks and caches used to suppress duplicate unsigned provenance warnings.
UNSIGNED_WARNING_LOCK = threading.Lock()
UNSIGNED_WARNING_CACHE: set[str] = set()

UNSIGNED_PROVENANCE_WARNING_LOCK = threading.Lock()
UNSIGNED_PROVENANCE_WARNING_CACHE: set[tuple[str, int | None]] = set()

# Track patch hashes that have already been emitted so that subsequent
# attempts to derive the same unsigned provenance do not keep printing the
# hash to stdout.
PATCH_HASH_LOCK = threading.Lock()
PATCH_HASH_CACHE: set[str] = set()

__all__ = [
    "UNSIGNED_WARNING_LOCK",
    "UNSIGNED_WARNING_CACHE",
    "UNSIGNED_PROVENANCE_WARNING_LOCK",
    "UNSIGNED_PROVENANCE_WARNING_CACHE",
    "PATCH_HASH_LOCK",
    "PATCH_HASH_CACHE",
]
