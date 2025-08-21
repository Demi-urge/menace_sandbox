from __future__ import annotations

from typing import Any, Mapping

from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST

try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # type: ignore

_VIOLATIONS = _me.Gauge(
    "patch_safety_violations_total",
    "Vectors skipped due to safety violations",
    labelnames=["type"],
)

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())


def check_patch_safety(
    meta: Mapping[str, Any],
    *,
    max_alert_severity: float = 1.0,
    max_alerts: int = 5,
    license_denylist: set[str] | None = None,
) -> bool:
    """Return ``True`` when metadata passes safety checks."""

    denylist = license_denylist or _DEFAULT_LICENSE_DENYLIST

    sev = meta.get("alignment_severity")
    if sev is not None:
        try:
            if float(sev) > max_alert_severity:
                _VIOLATIONS.labels("severity").inc()
                return False
        except Exception:
            pass

    alerts = meta.get("semantic_alerts")
    if alerts is not None:
        try:
            count = len(alerts) if isinstance(alerts, (list, tuple, set)) else 1
            if count > max_alerts:
                _VIOLATIONS.labels("alerts").inc()
                return False
        except Exception:
            pass

    lic = meta.get("license")
    fp = meta.get("license_fingerprint")
    if lic in denylist or _LICENSE_DENYLIST.get(fp) in denylist:
        _VIOLATIONS.labels("license").inc()
        return False

    return True


__all__ = ["check_patch_safety", "_VIOLATIONS"]
