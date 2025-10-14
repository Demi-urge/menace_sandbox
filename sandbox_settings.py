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

_FALLBACK_MODULE_CANDIDATES: tuple[str, ...] = (
    "menace_sandbox.sandbox_settings_fallback",
    "sandbox_settings_fallback",
)

_loaded: ModuleType | None = None
_last_error: Exception | None = None
_logged_fallback_notice = False
for dotted, attr in _MODULE_CANDIDATES:
    try:
        module = import_module(dotted)
    except ModuleNotFoundError as exc:
        _last_error = exc
        continue
    except Exception as exc:  # pragma: no cover - incompatible dependency versions
        _last_error = exc
        continue
    else:
        _loaded = module
        break

if _loaded is None:
    fallback_exc: ModuleNotFoundError | None = None
    for dotted in _FALLBACK_MODULE_CANDIDATES:
        try:
            module = import_module(dotted)
        except ModuleNotFoundError as exc:  # pragma: no cover - catastrophic
            fallback_exc = exc
            continue
        else:
            _loaded = module
            break
    if _loaded is None:
        exc = fallback_exc or ModuleNotFoundError(
            ", ".join(_FALLBACK_MODULE_CANDIDATES)
        )
        if _last_error is not None:
            exc.__cause__ = _last_error
        raise exc
    if _last_error is not None and not _logged_fallback_notice:
        _LOGGER.warning(
            "pydantic unavailable; using lightweight sandbox settings fallback: %s",
            _last_error,
        )
        _logged_fallback_notice = True
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

