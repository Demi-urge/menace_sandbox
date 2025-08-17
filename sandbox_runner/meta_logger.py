from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from logging_utils import get_logger, log_record
from audit_trail import AuditTrail
try:  # optional dependency
    from relevancy_metrics_db import RelevancyMetricsDB
except Exception:  # pragma: no cover - optional
    RelevancyMetricsDB = None  # type: ignore

try:  # avoid heavy dependency during light imports
    from .cycle import _async_track_usage
except Exception:  # pragma: no cover - best effort stub
    def _async_track_usage(*_a, **_k) -> None:  # type: ignore
        pass

logger = get_logger(__name__)


@dataclass
class _CycleMeta:
    cycle: int
    roi: float
    delta: float
    modules: list[str]
    reason: str
    warnings: dict | None = None


class _SandboxMetaLogger:
    def __init__(
        self,
        path: Path,
        module_index: "ModuleIndexDB" | None = None,
        metrics_db: RelevancyMetricsDB | None = None,
    ) -> None:
        self.path = path
        self.audit = AuditTrail(str(path))
        self.records: list[_CycleMeta] = []
        self.module_deltas: dict[str, list[float]] = {}
        self.flagged_sections: set[str] = set()
        self.last_patch_id = 0
        self.module_index = module_index
        self.metrics_db = metrics_db
        logger.debug("SandboxMetaLogger initialised at %s", path)

    def log_cycle(
        self,
        cycle: int,
        roi: float,
        modules: list[str],
        reason: str,
        warnings: dict | None = None,
        *,
        exec_time: float = 0.0,
    ) -> None:
        prev = self.records[-1].roi if self.records else 0.0
        delta = roi - prev
        self.records.append(_CycleMeta(cycle, roi, delta, modules, reason, warnings))
        per_module_delta = delta / len(modules) if modules else 0.0
        for m in modules:
            if self.module_index:
                try:
                    from pathlib import Path

                    gid = str(self.module_index.get(Path(m).name))
                except Exception:
                    gid = m
            else:
                gid = m
            self.module_deltas.setdefault(gid, []).append(delta)
            if self.metrics_db:
                try:
                    self.metrics_db.record(
                        m,
                        exec_time,
                        self.module_index,
                        roi_delta=per_module_delta,
                    )
                except Exception:
                    logger.exception("relevancy metrics record failed")
            _async_track_usage(gid, per_module_delta)
        try:
            record = {
                "cycle": cycle,
                "roi": roi,
                "delta": delta,
                "modules": modules,
                "reason": reason,
            }
            if warnings:
                record["warnings"] = warnings
            self.audit.record(record)
        except Exception:
            logger.exception("meta log record failed")
        logger.debug(
            "cycle %d logged roi=%s delta=%s modules=%s", cycle, roi, delta, modules
        )

    def rankings(self) -> list[tuple[str, float]]:
        totals = {m: sum(v) for m, v in self.module_deltas.items()}
        logger.debug("rankings computed: %s", totals)
        return sorted(totals.items(), key=lambda x: x[1], reverse=True)

    def diminishing(
        self,
        threshold: float | None = None,
        consecutive: int = 3,
        entropy_flags: Sequence[str] | None = None,
    ) -> list[str]:
        flags: list[str] = []
        thr = 0.0 if threshold is None else float(threshold)
        eps = 1e-3
        for m, vals in self.module_deltas.items():
            if m in self.flagged_sections:
                continue
            if len(vals) < consecutive:
                continue
            window = vals[-consecutive:]
            mean = sum(window) / consecutive
            if len(window) > 1:
                var = sum((v - mean) ** 2 for v in window) / len(window)
                std = var ** 0.5
            else:
                std = 0.0
            if abs(mean) <= thr and std < eps:
                flags.append(m)
                self.flagged_sections.add(m)
        if entropy_flags:
            for m in entropy_flags:
                if m not in self.flagged_sections:
                    flags.append(m)
                    self.flagged_sections.add(m)
        if flags:
            logger.debug("modules with diminishing returns: %s", flags)
        return flags

    def flag_modules(self, modules: Sequence[str]) -> None:
        """Mark ``modules`` as completed and add them to ``flagged_sections``."""
        new_flags: list[str] = []
        for m in modules:
            if m not in self.flagged_sections:
                self.flagged_sections.add(m)
                new_flags.append(m)
        if not new_flags:
            return
        try:
            self.audit.record(
                {
                    "cycle": len(self.records),
                    "modules": new_flags,
                    "reason": "entropy_ceiling",
                }
            )
        except Exception:
            logger.exception("meta log flag record failed")
        logger.info("modules flagged complete", extra=log_record(modules=new_flags))


__all__ = ["_SandboxMetaLogger"]
