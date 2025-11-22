from __future__ import annotations

"""Utilities for logging sandbox execution metadata.

This module requires telemetry support provided by ``relevancy_radar``. Importing
will raise :class:`ImportError` if the telemetry dependency is unavailable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, Mapping, Any
import json
import math

from .logging_utils import get_logger, log_record
from dynamic_path_router import path_for_prompt
from audit_trail import AuditTrail

try:  # optional dependency
    from relevancy_metrics_db import RelevancyMetricsDB
except ImportError:  # pragma: no cover - optional
    RelevancyMetricsDB = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from module_index_db import ModuleIndexDB
    from .workflow_sandbox_runner import ModuleMetrics

try:  # avoid heavy dependency during light imports
    from .cycle import _async_track_usage
except Exception:  # pragma: no cover - best effort fallback
    def _async_track_usage(*_a: object, **_k: object) -> None:
        """Fallback when telemetry support is unavailable."""
        return None

logger = get_logger(__name__)


@dataclass
class _CycleMeta:
    cycle: int
    roi: float
    delta: float
    modules: list[str]
    reason: str
    warnings: dict | None = None
    coverage: dict[str, float] | None = None
    coverage_summary: dict | None = None
    duration: float = 0.0
    errors: dict[str, str] | None = None
    successes: int = 0
    failures: int = 0
    coverage_percent: float = 0.0


class _SandboxMetaLogger:
    def __init__(
        self,
        path: Path,
        module_index: "ModuleIndexDB" | None = None,
        metrics_db: RelevancyMetricsDB | None = None,
    ) -> None:
        self.path = path
        self.flags_path = path.with_suffix(path.suffix + ".flags")
        self.history_path = path.with_suffix(path.suffix + ".hist")
        self.audit = AuditTrail(str(path))
        self.records: list[_CycleMeta] = []
        self.module_deltas: dict[str, list[float]] = {}
        self.module_entropies: dict[str, list[float]] = {}
        self.module_entropy_deltas: dict[str, list[float]] = {}
        self.flagged_sections: set[str] = set()
        self.last_patch_id = 0
        self.module_index = module_index
        self.metrics_db = metrics_db
        if self.flags_path.exists():
            try:
                data = json.loads(self.flags_path.read_text())
                if isinstance(data, list):
                    self.flagged_sections.update(str(x) for x in data)
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - best effort
                logger.exception("failed to load sandbox flags", exc_info=exc)
        if self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text())
                self.module_deltas.update(
                    {str(k): list(map(float, v)) for k, v in data.get("module_deltas", {}).items()}
                )
                self.module_entropies.update(
                    {
                        str(k): list(map(float, v))
                        for k, v in data.get("module_entropies", {}).items()
                    }
                )
                for m, hist in self.module_entropies.items():
                    if hist:
                        deltas = [0.0]
                        deltas.extend(
                            hist[i] - hist[i - 1] for i in range(1, len(hist))
                        )
                        self.module_entropy_deltas[m] = deltas
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - best effort
                logger.exception("failed to load sandbox history", exc_info=exc)
        logger.debug(
            "SandboxMetaLogger initialised at %s", path_for_prompt(path)
        )

    def log_cycle(
        self,
        cycle: int,
        roi: float,
        modules: list[str],
        reason: str,
        warnings: dict | None = None,
        *,
        exec_time: float = 0.0,
        module_metrics: Sequence["ModuleMetrics"] | None = None,
        coverage: Mapping[str, float] | None = None,
        coverage_summary: Mapping[str, Any] | None = None,
        duration: float | None = None,
        errors: Mapping[str, str] | None = None,
        successes: int | None = None,
        failures: int | None = None,
        coverage_percent: float | None = None,
    ) -> None:
        prev = self.records[-1].roi if self.records else 0.0
        delta = roi - prev
        cov: dict[str, float] | None = dict(coverage) if coverage else None
        cov_sum: dict | None = dict(coverage_summary) if coverage_summary else None
        err: dict[str, str] | None = dict(errors) if errors else None
        dur = float(duration or 0.0)
        succ = int(successes or 0)
        fail = int(failures or 0)
        cov_pct = float(coverage_percent or 0.0)
        gid_map: dict[str, str] = {}
        per_module_delta = delta / len(modules) if modules else 0.0
        for m in modules:
            if self.module_index:
                try:
                    from pathlib import Path

                    gid = str(self.module_index.get(Path(m).name))
                except Exception as exc:
                    logger.warning(
                        "module index lookup failed for %s: %s", m, exc
                    )
                    gid = m
            else:
                gid = m
            gid_map[m] = gid
            self.module_deltas.setdefault(gid, []).append(per_module_delta)
            self.entropy_delta(gid)
            if self.metrics_db:
                try:
                    self.metrics_db.record(
                        m,
                        exec_time,
                        self.module_index,
                        roi_delta=per_module_delta,
                    )
                except Exception as exc:
                    logger.exception(
                        "relevancy metrics record failed", exc_info=exc
                    )
            _async_track_usage(gid, per_module_delta)

        if module_metrics:
            cov = cov or {}
            err = err or {}
            covered = 0
            for m in module_metrics:
                gid = gid_map.get(m.name, m.name)
                if getattr(m, "entropy_delta", None) is not None:
                    self.module_entropy_deltas.setdefault(gid, []).append(
                        float(m.entropy_delta)
                    )
                if m.coverage_functions or m.coverage_files:
                    total = len(m.coverage_functions or []) + len(m.coverage_files or [])
                    cov[m.name] = float(total)
                    covered += 1
                if not m.success and m.exception:
                    err[m.name] = m.exception
                else:
                    succ += int(m.success)
                fail += int(not m.success)
                dur += m.duration
            if module_metrics and not cov_pct and len(module_metrics) > 0:
                cov_pct = (covered / len(module_metrics)) * 100.0
            if not cov:
                cov = None
            if not err:
                err = None

        self.records.append(
            _CycleMeta(
                cycle,
                roi,
                delta,
                modules,
                reason,
                warnings,
                cov,
                cov_sum,
                dur,
                err,
                succ,
                fail,
                cov_pct,
            )
        )
        self._persist_history()
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
            if cov:
                record["coverage"] = cov
            if cov_sum:
                record["coverage_summary"] = cov_sum
            if dur:
                record["duration"] = dur
            if err:
                record["errors"] = err
            if succ or fail:
                record["successes"] = succ
                record["failures"] = fail
            if cov_pct:
                record["coverage_percent"] = cov_pct
            self.audit.record(record)
        except Exception as exc:
            logger.exception("meta log record failed", exc_info=exc)
        logger.debug(
            "cycle %d logged roi=%s delta=%s modules=%s", cycle, roi, delta, modules
        )

    def rankings(self) -> list[tuple[int, float, int, int, float, float]]:
        rows: list[tuple[int, float, int, int, float, float]] = []
        for rec in self.records:
            rows.append(
                (
                    rec.cycle,
                    rec.roi,
                    rec.successes,
                    rec.failures,
                    rec.duration,
                    rec.coverage_percent,
                )
            )
        logger.debug("rankings computed: %s", rows)
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def entropy_delta(self, module: str, window: int = 5) -> float:
        vals = self.module_deltas.get(module)
        if not vals:
            return 0.0
        win = vals[-window:]
        total = sum(abs(v) for v in win)
        if total <= 0:
            entropy = 0.0
        else:
            entropy = -sum(
                (abs(v) / total) * math.log(abs(v) / total, 2)
                for v in win if v != 0
            )
        hist = self.module_entropies.setdefault(module, [])
        prev = hist[-1] if hist else entropy
        hist.append(entropy)
        delta = entropy - prev
        self.module_entropy_deltas.setdefault(module, []).append(delta)
        return delta

    def _persist_history(self) -> None:
        try:
            data = {
                "module_deltas": self.module_deltas,
                "module_entropies": self.module_entropies,
            }
            self.history_path.write_text(json.dumps(data))
        except (OSError, TypeError) as exc:  # pragma: no cover - best effort
            logger.exception("failed to persist sandbox history", exc_info=exc)

    def ceiling(self, ratio_threshold: float, consecutive: int = 3) -> list[str]:
        """Return modules where ROI gain per entropy delta diminishes."""

        flags: list[str] = []
        thr = float(ratio_threshold)
        for m, roi_vals in self.module_deltas.items():
            if m in self.flagged_sections:
                continue
            ent_vals = self.module_entropy_deltas.get(m)
            if not ent_vals:
                continue
            ratios = [
                abs(r) / abs(e) if e != 0 else float("inf")
                for r, e in zip(roi_vals, ent_vals)
                if e != 0
            ]
            if len(ratios) < consecutive:
                continue
            avgs = [
                sum(ratios[i - consecutive + 1: i + 1]) / consecutive
                for i in range(consecutive - 1, len(ratios))
            ]
            if avgs and all(avg < thr for avg in avgs[-consecutive:]):
                flags.append(m)
        if flags:
            self.flag_modules(flags, reason="entropy_ceiling")
            logger.debug("modules hitting entropy ceiling: %s", flags)
        return flags

    def diminishing(
        self,
        threshold: float | None = None,
        consecutive: int = 3,
        entropy_threshold: float | None = None,
    ) -> list[str]:
        flags: list[str] = []
        thr = 0.0 if threshold is None else float(threshold)
        e_thr = thr if entropy_threshold is None else float(entropy_threshold)
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
            roi_plateau = abs(mean) <= thr and std < eps

            ent_vals = self.module_entropy_deltas.get(m, [])
            if len(ent_vals) < len(vals):
                try:
                    self.entropy_delta(m)
                except Exception as exc:
                    logger.exception(
                        "entropy delta computation failed for %s", m, exc_info=exc
                    )
                ent_vals = self.module_entropy_deltas.get(m, [])
            ent_plateau = False
            if len(ent_vals) >= consecutive:
                ent_plateau = all(
                    abs(v) <= e_thr for v in ent_vals[-consecutive:]
                )
            if roi_plateau or ent_plateau:
                flags.append(m)
        if flags:
            self.flag_modules(flags, reason="diminishing_returns")
            logger.debug("modules with diminishing returns: %s", flags)
        return flags

    def _persist_flags(self) -> None:
        try:
            self.flags_path.write_text(json.dumps(sorted(self.flagged_sections)))
        except OSError as exc:  # pragma: no cover - best effort
            logger.exception("failed to persist sandbox flags", exc_info=exc)

    def flag_modules(self, modules: Sequence[str], *, reason: str = "entropy_ceiling") -> None:
        """Mark ``modules`` as completed and add them to ``flagged_sections``."""
        new_flags = [m for m in modules if m not in self.flagged_sections]
        if not new_flags:
            return
        self.flagged_sections.update(new_flags)
        self._persist_flags()
        try:
            self.audit.record(
                {
                    "cycle": len(self.records),
                    "modules": new_flags,
                    "reason": reason,
                }
            )
        except Exception as exc:
            logger.exception("meta log flag record failed", exc_info=exc)
        logger.info("modules flagged complete", extra=log_record(modules=new_flags))


__all__ = ["_SandboxMetaLogger"]
