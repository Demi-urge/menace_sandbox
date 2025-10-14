from __future__ import annotations

"""Dependency health tracking utilities for the autonomous sandbox.

The autonomous sandbox has historically emitted ad-hoc log messages whenever an
optional dependency was missing.  In minimal execution environments this caused
long streams of warnings during startup, obscuring actionable issues and making
it difficult for orchestration tooling to determine whether a launch was
healthy.

This module provides a structured registry that callers can use to record
missing or restored dependencies together with remediation hints.  Consumers
(such as the ``--health-check`` command) can query the registry for a snapshot
of the dependency state and surface the information in a machine-friendly
format.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from typing import Any, Dict, Iterable, Mapping


class DependencyCategory(str, Enum):
    """Categories used when reporting dependency health."""

    PYTHON = "python"
    SYSTEM = "system"
    SERVICE = "service"
    DATA = "data"
    OTHER = "other"


class DependencySeverity(str, Enum):
    """Severity hint for dependency issues."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def to_logging_level(self) -> int:
        if self is DependencySeverity.CRITICAL:
            return logging.ERROR
        if self is DependencySeverity.WARNING:
            return logging.WARNING
        return logging.INFO


@dataclass(slots=True)
class DependencyStatus:
    """Represents the latest known state for a dependency."""

    name: str
    category: DependencyCategory
    available: bool
    optional: bool
    severity: DependencySeverity
    description: str | None = None
    reason: str | None = None
    remedy: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "available": self.available,
            "optional": self.optional,
            "severity": self.severity.value,
            "description": self.description,
            "reason": self.reason,
            "remedy": self.remedy,
            "metadata": dict(self.metadata),
        }


class DependencyHealthRegistry:
    """Thread-safe registry tracking dependency availability."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._statuses: Dict[str, DependencyStatus] = {}
        self._logged_missing: set[str] = set()

    def _key(self, name: str, category: DependencyCategory) -> str:
        return f"{category.value}:{name.lower()}"

    def mark_available(
        self,
        *,
        name: str,
        category: DependencyCategory,
        optional: bool,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> DependencyStatus:
        """Record that ``name`` is now available."""

        status = DependencyStatus(
            name=name,
            category=category,
            available=True,
            optional=optional,
            severity=DependencySeverity.INFO,
            description=description,
            metadata=dict(metadata or {}),
        )
        key = self._key(name, category)
        with self._lock:
            previous = self._statuses.get(key)
            self._statuses[key] = status
            if key in self._logged_missing and status.available:
                self._logged_missing.discard(key)
        if previous is not None and not previous.available:
            log = logger or logging.getLogger(__name__)
            log.info(
                "Dependency %s (%s) restored", name, category.value
            )
        return status

    def mark_missing(
        self,
        *,
        name: str,
        category: DependencyCategory,
        optional: bool,
        severity: DependencySeverity,
        description: str | None = None,
        reason: str | None = None,
        remedy: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
        log_once: bool = True,
    ) -> DependencyStatus:
        """Record that ``name`` is currently missing."""

        status = DependencyStatus(
            name=name,
            category=category,
            available=False,
            optional=optional,
            severity=severity,
            description=description,
            reason=reason,
            remedy=remedy,
            metadata=dict(metadata or {}),
        )
        key = self._key(name, category)
        with self._lock:
            self._statuses[key] = status
            already_logged = key in self._logged_missing
            if log_once and not already_logged:
                self._logged_missing.add(key)
        log = logger or logging.getLogger(__name__)
        if not log_once or not already_logged:
            message = (
                "Dependency %s (%s) missing" % (name, category.value)
            )
            if reason:
                message += f": {reason}"
            log.log(status.severity.to_logging_level(), message)
            if remedy:
                log.log(status.severity.to_logging_level(), "Remediation: %s", remedy)
        return status

    def summary(self, *, include_optional: bool = True) -> Dict[str, Any]:
        """Return a serialisable snapshot of the dependency state."""

        with self._lock:
            statuses = list(self._statuses.values())
        missing = [
            status.as_dict()
            for status in statuses
            if not status.available
            and (include_optional or not status.optional)
        ]
        available = [
            status.as_dict()
            for status in statuses
            if status.available
            and (include_optional or not status.optional)
        ]
        optional_missing = [
            status.as_dict()
            for status in statuses
            if not status.available and status.optional
        ]
        counts = {
            "total": len(statuses),
            "available": len(available),
            "missing": len(missing),
            "optional_missing": len(optional_missing),
        }
        return {
            "counts": counts,
            "missing": missing,
            "optional_missing": optional_missing if include_optional else [],
            "available": available,
        }

    def iter_missing(self) -> Iterable[DependencyStatus]:
        with self._lock:
            return [status for status in self._statuses.values() if not status.available]


dependency_registry = DependencyHealthRegistry()

__all__ = [
    "DependencyCategory",
    "DependencySeverity",
    "DependencyStatus",
    "DependencyHealthRegistry",
    "dependency_registry",
]
