from __future__ import annotations

"""Runtime loader for sandbox settings with optional dependency fallbacks."""

from importlib import import_module
import logging
from types import ModuleType

_LOGGER = logging.getLogger(__name__)

_MODULE_CANDIDATES: tuple[tuple[str, str | None], ...] = (
    ("menace_sandbox.sandbox_settings_pydantic", None),
    ("sandbox_settings_pydantic", None),
)

_FALLBACK_MODULE = "menace_sandbox.sandbox_settings_fallback"

_loaded: ModuleType | None = None
_last_error: Exception | None = None
for dotted, attr in _MODULE_CANDIDATES:
    try:
        module = import_module(dotted)
    except ModuleNotFoundError as exc:
        _last_error = exc
        continue
    else:
        _loaded = module
        break

if _loaded is None:
    try:
        module = import_module(_FALLBACK_MODULE)
    except ModuleNotFoundError as exc:  # pragma: no cover - catastrophic
        if _last_error is not None:
            exc.__cause__ = _last_error
        raise
    _loaded = module
    if _last_error is not None:
        _LOGGER.warning(
            "pydantic unavailable; using lightweight sandbox settings fallback.",
            exc_info=_last_error,
        )
else:
    if _last_error is not None:
        _LOGGER.debug(
            "sandbox settings loaded after initial import error", exc_info=_last_error
        )

globals().update({k: getattr(_loaded, k) for k in getattr(_loaded, "__all__", [])})

# Preserve metadata for introspection utilities relying on these module globals.
__all__ = getattr(_loaded, "__all__", [])
__doc__ = getattr(_loaded, "__doc__", None)
_getter = getattr(_loaded, "__getattr__", None)
if callable(_getter):  # pragma: no cover - delegation path
    __getattr__ = _getter  # type: ignore[assignment]
globals()["USING_SANDBOX_SETTINGS_FALLBACK"] = getattr(
    _loaded, "USING_SANDBOX_SETTINGS_FALLBACK", False
)

