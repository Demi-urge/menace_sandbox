"""Utility script to unblock self-coding bootstrap loops.

Running this script performs two actions:

1. Purge stale state and lock files that can leave the autonomous sandbox in a
   perpetual retry loop after a crashed or aborted bootstrap attempt.
2. Invoke :func:`internalize_coding_bot` for the requested bot so the
   :class:`~menace_sandbox.self_coding_manager.SelfCodingManager` is registered
   immediately instead of waiting for the registry back-off cycle.

It is intentionally lightweight so that it can be run from PowerShell or bash
without requiring any additional dependencies beyond the project itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent


# ``bootstrap_self_coding.py`` is often executed directly via ``python`` or
# ``py`` from the repository root on Windows.  In that scenario the process
# working directory is ``menace_sandbox`` itself, so ``sys.path`` only contains
# the package directory rather than its parent.  Absolute imports such as
# ``menace_sandbox.self_coding_manager`` therefore fail because Python expects
# ``sys.path`` entries to point to the *parent* of the package.  Insert the
# parent directory explicitly so that the package-style import works regardless
# of how the script is invoked.
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))


STALE_STATE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "sandbox_data" / "self_coding_engine_state.json",
    REPO_ROOT / "vector_service" / "failed_tags.json",
)

JOURNAL_FILES: tuple[Path, ...] = (
    REPO_ROOT / "logs" / "audit" / "audit_log.db-journal",
    REPO_ROOT / "logs" / "audit_log.db-shm",
    REPO_ROOT / "audit" / "audit.db-journal",
)


def _iter_cleanup_targets() -> Iterable[Path]:
    for path in STALE_STATE_FILES:
        yield path
        yield Path(f"{path}.lock")
    yield from JOURNAL_FILES


def _purge_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            LOGGER.info("Removed stale file: %s", path)
    except IsADirectoryError:
        return
    except OSError:
        LOGGER.warning("Failed to remove stale file: %s", path, exc_info=True)


def purge_stale_files() -> None:
    """Delete well-known stale lock and journal files if they exist."""

    for candidate in _iter_cleanup_targets():
        _purge_path(candidate)


def bootstrap_self_coding(bot_name: str) -> None:
    """Trigger :func:`internalize_coding_bot` for *bot_name*."""

    from menace_sandbox.bot_registry import BotRegistry
    from menace_sandbox.code_database import CodeDB
    from menace_sandbox.context_builder_util import create_context_builder
    from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
    from menace_sandbox.menace_memory_manager import MenaceMemoryManager
    from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
    from menace_sandbox.self_coding_engine import SelfCodingEngine
    from menace_sandbox.self_coding_manager import internalize_coding_bot
    from menace_sandbox.self_coding_thresholds import get_thresholds

    LOGGER.info("Bootstrapping self-coding manager for bot %s", bot_name)
    builder = create_context_builder()
    registry = BotRegistry()
    data_bot = DataBot(start_server=False)
    engine = SelfCodingEngine(
        CodeDB(),
        MenaceMemoryManager(),
        context_builder=builder,
    )
    pipeline = ModelAutomationPipeline(context_builder=builder)

    roi_threshold = error_threshold = test_failure_threshold = None
    try:
        thresholds = get_thresholds(bot_name)
    except Exception as exc:  # pragma: no cover - diagnostics only
        LOGGER.warning("Failed to load thresholds for %s: %s", bot_name, exc)
    else:
        roi_threshold = thresholds.roi_drop
        error_threshold = thresholds.error_increase
        test_failure_threshold = thresholds.test_failure_increase
        try:
            persist_sc_thresholds(
                bot_name,
                roi_drop=roi_threshold,
                error_increase=error_threshold,
                test_failure_increase=test_failure_threshold,
            )
        except Exception:  # pragma: no cover - best effort persistence
            LOGGER.exception("Failed to persist thresholds for %s", bot_name)

    manager = internalize_coding_bot(
        bot_name=bot_name,
        engine=engine,
        pipeline=pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        roi_threshold=roi_threshold,
        error_threshold=error_threshold,
        test_failure_threshold=test_failure_threshold,
    )
    if manager is None:
        raise RuntimeError(
            "internalize_coding_bot returned None; self-coding manager registration failed"
        )
    LOGGER.info("Self-coding manager registered: \"%s\"", type(manager).__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot-name",
        default="InformationSynthesisBot",
        help="Name of the bot to internalize (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip removal of stale state and lock files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.skip_cleanup:
        purge_stale_files()
    else:
        LOGGER.info("Skipping stale file cleanup as requested")

    bootstrap_self_coding(args.bot_name)


if __name__ == "__main__":
    main()

