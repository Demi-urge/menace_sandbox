"""Runtime customisations applied on interpreter startup.

In addition to wiring package aliases, this module also patches ``python-dotenv``
to fail gracefully when it encounters malformed ``.env`` files.  Several entry
points load environment variables automatically and historically a single bad
line (for example ``export FOO=bar`` copied from a shell script) would cause the
entire application to abort.  The wrapper implemented here catches the
``DotenvParseError`` exception raised by the library and converts it into a
logged warning so that execution can proceed using any successfully parsed
values.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

base = Path(__file__).resolve().parent
pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(base / "sandbox_runner")]
root = types.ModuleType("menace_sandbox")
root.__path__ = [str(base)]
sys.modules.setdefault("sandbox_runner", pkg)
sys.modules.setdefault("menace_sandbox", root)
sys.modules.setdefault("menace_sandbox.sandbox_runner", pkg)


def _patch_dotenv() -> None:
    """Install tolerant wrappers around ``python-dotenv`` helpers.

    ``python-dotenv`` raises :class:`~dotenv.exceptions.DotenvParseError` when it
    encounters unexpected syntax.  This is correct behaviour for strict setups
    but undesirable for the sandbox where environment files are often edited by
    hand.  We patch both :func:`dotenv.load_dotenv` and
    :func:`dotenv.dotenv_values` to log and continue instead of bubbling the
    exception.
    """

    try:
        import dotenv
        from dotenv.exceptions import DotenvParseError
    except Exception:  # pragma: no cover - optional dependency
        return

    logger = logging.getLogger("menace_sandbox.dotenv")

    original_load = getattr(dotenv, "load_dotenv", None)
    if callable(original_load):

        def _safe_load_dotenv(*args, **kwargs):
            try:
                return original_load(*args, **kwargs)
            except DotenvParseError as exc:  # pragma: no cover - requires bad file
                logger.warning("Failed to parse dotenv file: %s", exc)
                return False

        dotenv.load_dotenv = _safe_load_dotenv  # type: ignore[assignment]

    original_values = getattr(dotenv, "dotenv_values", None)
    if callable(original_values):

        def _safe_dotenv_values(*args, **kwargs):
            try:
                return original_values(*args, **kwargs)
            except DotenvParseError as exc:  # pragma: no cover - requires bad file
                logger.warning("Failed to parse dotenv file: %s", exc)
                return {}

        dotenv.dotenv_values = _safe_dotenv_values  # type: ignore[assignment]


_patch_dotenv()

