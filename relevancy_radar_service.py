from __future__ import annotations

"""Background service for running the :class:`RelevancyRadar` periodically."""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Iterable
import os

try:  # pragma: no cover - prefer package import
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - allow running as script
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - prefer package import but allow direct execution
    from .relevancy_radar import RelevancyRadar
except Exception:  # pragma: no cover - executed when run as a script
    from relevancy_radar import RelevancyRadar


class RelevancyRadarService:
    """Periodically scan for unused modules and trigger retirements."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        interval: int = 3600,
        event_bus: "UnifiedEventBus | None" = None,
    ) -> None:
        self.root = Path(repo_root)
        self.interval = interval
        self.logger = logging.getLogger(self.__class__.__name__)
        self._thread: threading.Thread | None = None
        self.running = False
        self.latest_flags: Dict[str, str] = {}
        self._retirement_service: ModuleRetirementService | None = None

        if event_bus is None:
            try:
                from .unified_event_bus import UnifiedEventBus
            except Exception:  # pragma: no cover - executed when run as script
                from unified_event_bus import UnifiedEventBus
            event_bus = UnifiedEventBus()
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    def _modules(self) -> Iterable[str]:
        """Return an iterable of module identifiers under ``repo_root``."""
        try:
            try:
                from .dynamic_module_mapper import build_module_map
            except Exception:  # pragma: no cover - executed when run as script
                from dynamic_module_mapper import build_module_map

            mapping = build_module_map(self.root)
            return mapping
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("module map build failed")
            return []

    def _scan_once(self) -> None:
        try:
            try:
                from .module_graph_analyzer import build_import_graph
                from .relevancy_radar import load_usage_stats
                from .relevancy_metrics_db import RelevancyMetricsDB
                from .metrics_exporter import (
                    update_relevancy_metrics,
                    relevancy_flags_retire_total,
                    relevancy_flags_compress_total,
                    relevancy_flags_replace_total,
                )
                from .unified_event_bus import UnifiedEventBus
            except Exception:  # pragma: no cover - executed when run as script
                from module_graph_analyzer import build_import_graph
                from relevancy_radar import load_usage_stats
                from relevancy_metrics_db import RelevancyMetricsDB
                from metrics_exporter import (
                    update_relevancy_metrics,
                    relevancy_flags_retire_total,
                    relevancy_flags_compress_total,
                    relevancy_flags_replace_total,
                )
                from unified_event_bus import UnifiedEventBus

            try:
                from .sandbox_settings import SandboxSettings
            except Exception:  # pragma: no cover - executed when run as script
                from sandbox_settings import SandboxSettings

            try:
                settings = SandboxSettings()
                compress = float(settings.relevancy_radar_compress_ratio)
                replace = float(settings.relevancy_radar_replace_ratio)
            except Exception:
                compress, replace = 0.01, 0.05

            graph = build_import_graph(self.root)
            module_names = {n.replace("/", ".") for n in graph.nodes}

            radar = RelevancyRadar()
            existing_metrics = radar._load_metrics()
            radar._metrics.clear()

            usage_stats = load_usage_stats()
            module_names.update(usage_stats.keys())

            try:
                data_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
                db = RelevancyMetricsDB(data_dir / "relevancy_metrics.db")
                roi_deltas = db.get_roi_deltas(module_names)
            except Exception:
                roi_deltas = {}

            for mod in module_names:
                count = int(usage_stats.get(mod, 0))
                prev = existing_metrics.get(mod, {})
                info = {
                    "imports": count,
                    "executions": count,
                    "impact": float(prev.get("impact", 0.0))
                    + float(roi_deltas.get(mod, 0.0)),
                    "output_impact": float(prev.get("output_impact", 0.0)),
                }
                radar._metrics[mod] = info

            radar._persist_metrics()

            import inspect

            evaluator = getattr(radar, "evaluate_final_contribution")
            if "graph" in inspect.signature(evaluator).parameters:
                flags = evaluator(
                    compress,
                    replace,
                    graph=graph,
                    core_modules=["menace_master", "run_autonomous"],
                )
            else:
                flags = radar.evaluate_relevance(
                    compress,
                    replace,
                    dep_graph=graph,
                    core_modules=["menace_master", "run_autonomous"],
                )

            self.latest_flags = flags
            update_relevancy_metrics(flags)

            event_bus = self._event_bus
            if event_bus is None:
                event_bus = UnifiedEventBus()
                self._event_bus = event_bus

            from collections import Counter

            counts = Counter(flags.values())
            if counts.get("retire"):
                relevancy_flags_retire_total.inc(float(counts["retire"]))
            if counts.get("compress"):
                relevancy_flags_compress_total.inc(float(counts["compress"]))
            if counts.get("replace"):
                relevancy_flags_replace_total.inc(float(counts["replace"]))
            if flags:
                try:
                    event_bus.publish("relevancy_flags", flags)
                except Exception:
                    self.logger.exception("failed to publish relevancy flags")
                if self._retirement_service is None:
                    try:
                        from .module_retirement_service import ModuleRetirementService
                    except Exception:  # pragma: no cover - executed when run as script
                        from module_retirement_service import ModuleRetirementService

                    self._retirement_service = ModuleRetirementService(self.root)
                self._retirement_service.process_flags(flags)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("relevancy scan failed")

    def _loop(self) -> None:
        while self.running:
            time.sleep(self.interval)
            self._scan_once()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        # Run an immediate scan so the service has up-to-date data before the
        # background thread kicks in.
        self.logger.info("performing initial relevancy scan")
        self._scan_once()
        self.logger.info("initial scan produced %d flags", len(self.latest_flags))

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None

    def flags(self) -> Dict[str, str]:
        """Return the most recent set of relevancy flags."""
        return dict(self.latest_flags)


__all__ = ["RelevancyRadarService"]
