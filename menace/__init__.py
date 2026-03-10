"""Compatibility package forwarding imports to project root."""

from __future__ import annotations

from pathlib import Path

from menace_sandbox.legacy_symbol_compat import resolve_legacy_symbol

try:  # pragma: no cover - optional helper module
    from dynamic_path_router import repo_root as _repo_root
except ModuleNotFoundError:  # pragma: no cover - fallback when module missing
    def repo_root() -> Path:
        """Return the repository root when :mod:`dynamic_path_router` is absent."""

        return Path(__file__).resolve().parent.parent
else:
    repo_root = _repo_root

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

_LEGACY_SYMBOL_ALIASES = {
    "audit_logger": "audit_logger",
    "self_debugger_sandbox": "self_debugger_sandbox",
    "patch_generator": "menace_sandbox.patch_generator",
    "sandbox_runner": "sandbox_runner",
}


def __getattr__(name: str) -> object:
    if name in _LEGACY_SYMBOL_ALIASES:
        resolved = resolve_legacy_symbol(
            symbol=name,
            aliases=_LEGACY_SYMBOL_ALIASES,
            package_name=__name__,
        )
        globals()[name] = resolved
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RAISE_ERRORS", "MenaceDB", "NUMERIC_BACKEND", *sorted(_LEGACY_SYMBOL_ALIASES)]
