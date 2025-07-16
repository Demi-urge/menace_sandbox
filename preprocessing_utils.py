from __future__ import annotations

"""Utilities for preparing datasets."""

from typing import Any, Iterable, Mapping, Dict, List

from .neuroplasticity import Outcome


def normalize_features(X: List[List[float]]) -> List[List[float]]:
    """Return column-wise normalized copy of ``X``."""
    if not X:
        return X
    cols = list(zip(*X))
    mins = [min(c) for c in cols]
    maxs = [max(c) for c in cols]
    norm = []
    for row in X:
        norm.append([
            (v - mi) / (ma - mi) if ma != mi else 0.0
            for v, mi, ma in zip(row, mins, maxs)
        ])
    return norm

def fill_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` or *default* when invalid."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def encode_outcome(outcome: str | Outcome) -> List[int]:
    """One-hot encode an :class:`Outcome` value."""
    if not isinstance(outcome, Outcome):
        try:
            outcome = Outcome(str(outcome).upper())
        except Exception:
            outcome = Outcome.FAILURE
    return [
        int(outcome is Outcome.SUCCESS),
        int(outcome is Outcome.PARTIAL_SUCCESS),
        int(outcome is Outcome.FAILURE),
    ]


def validate_memory_record(rec: Mapping[str, Any]) -> bool:
    """Return ``True`` if *rec* looks like a valid memory record."""
    return bool(rec.get("key")) and rec.get("data") is not None


def clean_memory_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalise memory records."""
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not validate_memory_record(rec):
            continue
        item = dict(rec)
        item["version"] = fill_float(item.get("version"))
        item["tags"] = item.get("tags") or ""
        cleaned.append(item)
    return cleaned


def validate_code_record(rec: Mapping[str, Any]) -> bool:
    return bool(rec.get("code"))


def clean_code_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not validate_code_record(rec):
            continue
        item = dict(rec)
        item["complexity_score"] = fill_float(item.get("complexity_score"))
        cleaned.append(item)
    return cleaned


def validate_roi_record(rec: Mapping[str, Any]) -> bool:
    """Return True if *rec* has minimal ROI fields."""
    return rec.get("bot") is not None


def clean_roi_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not validate_roi_record(rec):
            continue
        item = dict(rec)
        item["revenue"] = fill_float(item.get("revenue"))
        item["api_cost"] = fill_float(item.get("api_cost"))
        item["cpu_seconds"] = fill_float(item.get("cpu_seconds"))
        item["success_rate"] = fill_float(item.get("success_rate"))
        cleaned.append(item)
    return cleaned


def validate_workflow_record(rec: Mapping[str, Any]) -> bool:
    return bool(rec.get("workflow"))


def clean_workflow_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not validate_workflow_record(rec):
            continue
        item = dict(rec)
        item["status"] = item.get("status") or ""
        item["tags"] = item.get("tags") or []
        cleaned.append(item)
    return cleaned


__all__ = [
    "normalize_features",
    "fill_float",
    "encode_outcome",
    "validate_memory_record",
    "clean_memory_records",
    "validate_code_record",
    "clean_code_records",
    "validate_roi_record",
    "clean_roi_records",
    "validate_workflow_record",
    "clean_workflow_records",
]
