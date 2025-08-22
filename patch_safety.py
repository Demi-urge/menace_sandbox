from __future__ import annotations

"""Assess patch safety via metadata filters and failure embeddings.

The original version of this module only tracked failure embeddings and
exposed :class:`PatchSafety` for similarity scoring.  The vector service also
implemented a separate ``check_patch_safety`` helper that rejected vectors
based on metadata such as license fingerprints or semantic alert counts.

This file now consolidates both concerns.  ``PatchSafety`` provides an
``evaluate`` method which first applies the lightweight metadata checks and,
when available, scores the candidate against known failure embeddings.  The
return value is a tuple ``(passed, score)`` where ``passed`` indicates whether
the patch is acceptable and ``score`` is the similarity against recorded
failures.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from error_vectorizer import ErrorVectorizer

try:  # pragma: no cover - optional dependency for metrics
    from vector_service import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    try:  # type: ignore[attr-defined]
        import metrics_exporter as _me  # type: ignore
    except Exception:  # pragma: no cover - very defensive fallback
        class _Dummy:
            def Gauge(self, *_, **__):  # type: ignore
                class _Null:
                    def labels(self, *_, **__):  # pragma: no cover - noop
                        return self

                    def inc(self, *_: Any, **__: Any) -> None:  # pragma: no cover - noop
                        return None

                return _Null()

        _me = _Dummy()  # type: ignore


_VIOLATIONS = _me.Gauge(
    "patch_safety_violations_total",
    "Patches rejected due to safety violations",
    labelnames=["type"],
)

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())


def _cosine(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class PatchSafety:
    """Store failure embeddings and evaluate new patches."""

    threshold: float = 0.8
    max_alert_severity: float = 1.0
    max_alerts: int = 5
    license_denylist: set[str] = field(
        default_factory=lambda: set(_DEFAULT_LICENSE_DENYLIST)
    )
    vectorizer: ErrorVectorizer = field(default_factory=ErrorVectorizer)
    storage_path: str | None = None
    _failures: List[List[float]] = field(default_factory=list)
    _records: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # pragma: no cover - simple IO
        self.load_failures()

    # ------------------------------------------------------------------
    def record_failure(self, err: Dict[str, Any]) -> None:
        """Add a failure example represented by ``err`` and persist it."""
        self.vectorizer.fit([err])
        vec = self.vectorizer.transform(err)
        self._failures.append(vec)
        self._records.append(err)
        # best-effort persistence
        try:  # pragma: no cover - simple IO
            self.save_failures()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def load_failures(self, path: str | None = None) -> None:
        """Populate ``_failures`` and vectoriser state from ``path``."""
        pth = path or self.storage_path
        if not pth:
            return
        p = Path(pth)
        if not p.exists():
            return
        try:  # pragma: no cover - simple IO
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict):
                        err = data.get("err", {})
                        vec = data.get("vector")
                    else:
                        err = {}
                        vec = data
                    if err:
                        try:
                            self.vectorizer.fit([err])
                        except Exception:
                            pass
                        self._records.append(err)
                        if vec is None:
                            try:
                                vec = self.vectorizer.transform(err)
                            except Exception:
                                vec = []
                    if isinstance(vec, list):
                        self._failures.append(vec)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def save_failures(self, path: str | None = None) -> None:
        """Append the most recent failure to ``path``."""
        if not self._records or not self._failures:
            return
        pth = path or self.storage_path
        if not pth:
            return
        p = Path(pth)
        try:  # pragma: no cover - simple IO
            with p.open("a", encoding="utf-8") as fh:
                json.dump({"err": self._records[-1], "vector": self._failures[-1]}, fh)
                fh.write("\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def score(self, err: Dict[str, Any]) -> float:
        """Return the maximum similarity between ``err`` and recorded failures."""
        if not self._failures:
            return 0.0
        vec = self.vectorizer.transform(err)
        return max(_cosine(vec, f) for f in self._failures)

    # ------------------------------------------------------------------
    def _check_meta(self, meta: Mapping[str, Any]) -> bool:
        """Return ``True`` when metadata passes safety checks."""

        sev = meta.get("alignment_severity")
        if sev is not None:
            try:
                if float(sev) > self.max_alert_severity:
                    _VIOLATIONS.labels("severity").inc()
                    return False
            except Exception:  # pragma: no cover - defensive parsing
                pass

        alerts = meta.get("semantic_alerts")
        if alerts is not None:
            try:
                count = len(alerts) if isinstance(alerts, (list, tuple, set)) else 1
                if count > self.max_alerts:
                    _VIOLATIONS.labels("alerts").inc()
                    return False
            except Exception:  # pragma: no cover - defensive parsing
                pass

        lic = meta.get("license")
        fp = meta.get("license_fingerprint")
        denylist = self.license_denylist or _DEFAULT_LICENSE_DENYLIST
        if lic in denylist or _LICENSE_DENYLIST.get(fp) in denylist:
            _VIOLATIONS.labels("license").inc()
            return False

        return True

    # ------------------------------------------------------------------
    def evaluate(
        self, meta: Mapping[str, Any], err: Dict[str, Any] | None = None
    ) -> Tuple[bool, float]:
        """Return ``(passed, score)`` for ``meta`` and optional ``err``.

        ``passed`` is ``False`` when the metadata violates any denylist or when
        the similarity ``score`` exceeds the ``threshold``.
        """

        if not self._check_meta(meta):
            return False, 0.0

        score = 0.0
        if err is not None:
            try:
                score = float(self.score(err))
            except Exception:  # pragma: no cover - defensive
                score = 0.0
        passed = score < self.threshold
        return passed, score


def check_patch_safety(
    meta: Mapping[str, Any],
    *,
    max_alert_severity: float = 1.0,
    max_alerts: int = 5,
    license_denylist: set[str] | None = None,
) -> bool:
    """Backward compatible wrapper around :class:`PatchSafety`.

    This helper mirrors the behaviour of the previous implementation used in
    several parts of the code base.  It ignores similarity scoring and simply
    returns whether the metadata passes the safety checks.
    """

    ps = PatchSafety(
        max_alert_severity=max_alert_severity,
        max_alerts=max_alerts,
        license_denylist=set(license_denylist or _DEFAULT_LICENSE_DENYLIST),
    )
    passed, _ = ps.evaluate(meta)
    return passed


__all__ = ["PatchSafety", "check_patch_safety", "_VIOLATIONS"]
