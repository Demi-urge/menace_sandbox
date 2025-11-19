from __future__ import annotations

"""Utility to discover coding bots and register them with core services."""

from pathlib import Path
import importlib.util
import logging
import sys
from typing import Iterable

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)

try:  # optional import to satisfy type checkers at runtime
    from .bot_registry import BotRegistry
except Exception:  # pragma: no cover - fallback when run flat
    from bot_registry import BotRegistry  # type: ignore

try:
    from .threshold_service import threshold_service
except Exception:  # pragma: no cover - fallback when run flat
    from threshold_service import threshold_service  # type: ignore

try:  # optional import to satisfy type checkers at runtime
    from .evolution_orchestrator import EvolutionOrchestrator
except Exception:  # pragma: no cover - avoid hard dependency during startup
    try:  # pragma: no cover - fallback when executed as flat module
        from evolution_orchestrator import EvolutionOrchestrator  # type: ignore
    except Exception:  # pragma: no cover - degrade gracefully
        EvolutionOrchestrator = None  # type: ignore

logger = logging.getLogger(__name__)


def _iter_bot_modules(root: Path) -> Iterable[Path]:
    """Yield Python modules ending with ``_bot.py`` under *root*.

    Directories used for tests or documentation are ignored to avoid
    registering non-runtime helpers.
    """

    ignore = {"tests", "unit_tests", "docs"}
    for path in root.rglob("*_bot.py"):
        if any(part in ignore or part.startswith("test") for part in path.parts):
            continue
        yield path


def discover_and_register_coding_bots(
    registry: BotRegistry,
    orchestrator: EvolutionOrchestrator | None = None,
    *,
    root: Path | None = None,
) -> list[str]:
    """Scan *root* for coding bots and register them.

    Parameters
    ----------
    registry:
        :class:`BotRegistry` instance receiving bot registrations.
    orchestrator:
        Optional :class:`EvolutionOrchestrator` used to track metrics for each
        bot. When ``None`` only the registry is updated.
    root:
        Directory tree to search.  Defaults to the repository root determined
        from this file's location.

    Returns
    -------
    list[str]
        Names of bots that were successfully registered.
    """

    root_path = root or Path(__file__).resolve().parent
    registered: list[str] = []
    for mod_path in _iter_bot_modules(root_path):
        bot_name = mod_path.stem
        try:
            rt = threshold_service.reload(bot_name)
            registry.register_bot(
                bot_name,
                roi_threshold=rt.roi_drop,
                error_threshold=rt.error_threshold,
                test_failure_threshold=rt.test_failure_threshold,
                is_coding_bot=False,
            )
            if orchestrator is not None and hasattr(orchestrator, "register_bot"):
                orchestrator.register_bot(bot_name)
            registered.append(bot_name)
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to register bot %s", bot_name)
    return registered

__all__ = ["discover_and_register_coding_bots"]
