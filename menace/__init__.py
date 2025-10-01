"""Compatibility package forwarding imports to project root."""

from __future__ import annotations

from dynamic_path_router import repo_root

# Allow importing modules from repository root using ``menace`` prefix.
__path__.append(str(repo_root()))

try:  # pragma: no cover - optional numeric backend
    from .numeric_backend import NUMERIC_BACKEND
except ImportError as exc:  # pragma: no cover - degrade gracefully in minimal envs
    class _MissingNumericBackend:
        """Placeholder that surfaces the optional numeric backend dependency."""

        def __getattr__(self, name: str) -> None:
            raise ImportError(
                "Menace requires either PyTorch or NumPy for numeric operations; "
                "install menace_sandbox[numeric] to enable NUMERIC_BACKEND"
            ) from exc

    NUMERIC_BACKEND = _MissingNumericBackend()  # type: ignore[assignment]

# Default flag used by modules expecting it
RAISE_ERRORS = False

# Expose MenaceDB for tests and compatibility
try:  # pragma: no cover - optional dependency
    from .databases import MenaceDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore

__all__ = ["RAISE_ERRORS", "MenaceDB", "NUMERIC_BACKEND"]
