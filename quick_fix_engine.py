from __future__ import annotations

"""Automatically propose fixes for recurring errors."""

from pathlib import Path
import logging
import subprocess
from typing import Tuple, Iterable

from .error_bot import ErrorDB
from .self_coding_manager import SelfCodingManager
from .knowledge_graph import KnowledgeGraph


def generate_patch(module: str, engine: "SelfCodingEngine" | None = None) -> int | None:
    """Attempt a quick patch for *module* and return the patch id.

    Parameters
    ----------
    module:
        Target module path or module name without ``.py``.
    engine:
        Optional :class:`~self_coding_engine.SelfCodingEngine` instance.  If not
        provided, a minimal engine is instantiated on demand.  The function
        tolerates missing dependencies and simply returns ``None`` on failure.
    """

    logger = logging.getLogger("QuickFixEngine")
    path = Path(module)
    if path.suffix == "":
        path = path.with_suffix(".py")
    if not path.exists():
        logger.error("module not found: %s", module)
        return None

    if engine is None:
        try:  # pragma: no cover - heavy dependencies
            from .self_coding_engine import SelfCodingEngine
            from .code_database import CodeDB
            from .menace_memory_manager import MenaceMemoryManager

            engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager())
        except Exception as exc:  # pragma: no cover - optional deps
            logger.error("self coding engine unavailable: %s", exc)
            return None

    try:
        patch_id: int | None
        try:
            patch_id, _, _ = engine.apply_patch(path, "preemptive_fix")
        except AttributeError:
            engine.patch_file(path, "preemptive_fix")
            patch_id = None
        return patch_id
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.error("quick fix generation failed for %s: %s", module, exc)
        return None


class QuickFixEngine:
    """Analyse frequent errors and trigger small patches."""

    def __init__(
        self,
        error_db: ErrorDB,
        manager: SelfCodingManager,
        *,
        threshold: int = 3,
        graph: KnowledgeGraph | None = None,
        risk_threshold: float = 0.5,
    ) -> None:
        self.db = error_db
        self.manager = manager
        self.threshold = threshold
        self.graph = graph or KnowledgeGraph()
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _top_error(
        self, bot: str
    ) -> Tuple[str, str, dict[str, int], int] | None:
        try:
            info = self.db.top_error_module(bot)
        except Exception:
            return None
        if not info:
            return None
        etype, module, mods, count, _ = info
        return etype, module, mods, count

    def run(self, bot: str) -> None:
        """Attempt a quick patch for the most frequent error of ``bot``."""
        info = self._top_error(bot)
        if not info:
            return
        etype, module, mods, count = info
        if count < self.threshold:
            return
        path = Path(f"{module}.py")
        if not path.exists():
            return
        desc = f"quick fix {etype}"
        patch_id = None
        try:
            result = self.manager.run_patch(path, desc)
            patch_id = getattr(result, "patch_id", None)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.error("quick fix failed for %s: %s", bot, exc)
        tests_ok = True
        try:
            subprocess.run(["pytest", "-q"], check=True)
        except Exception as exc:
            tests_ok = False
            self.logger.error("quick fix validation failed: %s", exc)
        try:
            self.graph.add_telemetry_event(
                bot, etype, module, mods, patch_id=patch_id, resolved=tests_ok
            )
            self.graph.update_error_stats(self.db)
        except Exception as exc:
            self.logger.exception("telemetry update failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    def preemptive_patch_modules(
        self,
        modules: Iterable[tuple[str, float]],
        *,
        risk_threshold: float | None = None,
    ) -> None:
        """Proactively patch ``modules`` that exceed ``risk_threshold``.

        Parameters
        ----------
        modules:
            Iterable of ``(module, risk_score)`` pairs as produced by
            :meth:`~error_cluster_predictor.ErrorClusterPredictor.predict_high_risk_modules`.
        risk_threshold:
            Minimum risk score required to trigger a patch.  Defaults to the
            instance's ``risk_threshold`` attribute.
        """

        thresh = self.risk_threshold if risk_threshold is None else risk_threshold
        for module, risk in modules:
            if risk < thresh:
                continue
            path = Path(f"{module}.py")
            if not path.exists():
                continue
            patch_id = None
            try:
                result = self.manager.run_patch(path, "preemptive_patch")
                patch_id = getattr(result, "patch_id", None)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.error("preemptive patch failed for %s: %s", module, exc)
                try:
                    patch_id = generate_patch(module, getattr(self.manager, "engine", None))
                except Exception:
                    patch_id = None
            try:
                self.db.log_preemptive_patch(module, risk, patch_id)
            except Exception as exc:  # pragma: no cover - db issues
                self.logger.error("failed to record preemptive patch for %s: %s", module, exc)

    # ------------------------------------------------------------------
    def run_and_validate(self, bot: str) -> None:
        """Run :meth:`run` then execute the test suite."""
        self.run(bot)


__all__ = ["QuickFixEngine", "generate_patch"]
