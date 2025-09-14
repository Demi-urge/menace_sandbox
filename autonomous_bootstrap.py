from __future__ import annotations

"""Bootstrap the sandbox and launch a self-improvement cycle."""

import logging
import types

from sandbox_settings import SandboxSettings
from sandbox_runner.bootstrap import (
    bootstrap_environment,
    _verify_required_dependencies,
)
from self_improvement.api import init_self_improvement, start_self_improvement_cycle
from bot_registry import BotRegistry
from bot_discovery import discover_and_register_coding_bots


logger = logging.getLogger(__name__)


def main() -> int:
    """Bootstrap dependencies and run the self-improvement cycle."""
    logging.basicConfig(level=logging.INFO)
    logger.info("validating sandbox environment and dependencies")
    settings = SandboxSettings()
    try:
        settings = bootstrap_environment(settings, _verify_required_dependencies)
    except SystemExit as exc:
        logger.error("dependency verification failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected environment failure
        logger.exception("environment bootstrap failed")
        return 1

    try:
        init_self_improvement(settings)
    except Exception as exc:  # pragma: no cover - missing helper packages
        logger.exception("self-improvement initialisation failed")
        return 1

    # Discover coding bots so they participate in the self-improvement cycle
    registry = BotRegistry()
    orchestrator = None
    try:  # pragma: no cover - best effort orchestrator initialisation
        from evolution_orchestrator import EvolutionOrchestrator

        class _DB:
            def fetch(self, limit: int = 50):  # noqa: D401 - simple stub
                return []

        class _DataBot:
            def __init__(self) -> None:
                self.db = _DB()
                self.event_bus = None
                self.settings = None

            def check_degradation(self, *a, **k) -> None:  # pragma: no cover - stub
                return None

        class _Capital:
            def __init__(self, data_bot: _DataBot) -> None:  # pragma: no cover - stub
                self.data_bot = data_bot
                self.trend_predictor = None

        class _Improvement:
            def __init__(self, data_bot: _DataBot) -> None:  # pragma: no cover - stub
                self.data_bot = data_bot
                self.bot_name = ""

        data_bot = _DataBot()
        capital = _Capital(data_bot)
        improv = _Improvement(data_bot)
        manager = types.SimpleNamespace(bot_registry=registry, bot_name="", event_bus=None)
        orchestrator = EvolutionOrchestrator(
            data_bot,
            capital,
            improv,
            types.SimpleNamespace(),
            selfcoding_manager=manager,
        )
    except Exception as exc:  # pragma: no cover - orchestrator optional
        logger.warning("EvolutionOrchestrator unavailable: %s", exc)

    discover_and_register_coding_bots(registry, orchestrator)

    logger.info(
        "starting self-improvement cycle (repo=%s data_dir=%s)",
        settings.sandbox_repo_path,
        settings.sandbox_data_dir,
    )
    thread = start_self_improvement_cycle({"bootstrap": lambda: None})
    thread.start()
    logger.info("self-improvement cycle running; press Ctrl+C to stop")
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info("stopping self-improvement cycle")
        thread.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
