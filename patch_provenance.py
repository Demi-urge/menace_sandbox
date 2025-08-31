from __future__ import annotations

"""Helpers for querying patch vector ancestry information."""

from typing import Any, Dict, List, Mapping
import json
import logging

from code_database import PatchHistoryDB
from vector_service import PatchLogger


# ---------------------------------------------------------------------------
def get_patch_provenance(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return vectors influencing ``patch_id`` ordered by influence.

    Results are read from the ``patch_ancestry`` table and include the
    origin, influence score, detected license and any semantic alerts for each
    vector.
    """

    db = patch_db or PatchHistoryDB()
    rows = db.get_ancestry(patch_id)
    try:
        rec = db.get(patch_id)
        roi_before = float(getattr(rec, "roi_before", 0.0)) if rec else 0.0
        roi_after = float(getattr(rec, "roi_after", 0.0)) if rec else 0.0
        roi_delta = float(getattr(rec, "roi_delta", roi_after - roi_before))
        if roi_delta == 0.0:
            roi_delta = roi_after - roi_before
    except Exception:
        roi_before = roi_after = roi_delta = 0.0
    result: List[Dict[str, Any]] = []
    for row in rows:
        origin, vid, infl, *rest = row
        lic = rest[0] if len(rest) > 0 else None
        fp = rest[1] if len(rest) > 1 else None
        alerts = rest[2] if len(rest) > 2 else None
        result.append(
            {
                "origin": origin,
                "vector_id": vid,
                "influence": float(infl),
                "license": lic,
                "license_fingerprint": fp,
                "semantic_alerts": json.loads(alerts) if alerts else [],
                "roi_before": roi_before,
                "roi_after": roi_after,
                "roi_delta": roi_delta,
            }
        )
    return result


# ---------------------------------------------------------------------------
def get_roi_history(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> Dict[str, Any]:
    """Return stored ROI information for ``patch_id``.

    The result contains the overall ``roi_before``, ``roi_after`` and
    ``roi_delta`` values along with any per-origin ``roi_deltas`` mapping
    recorded during :func:`PatchLogger.track_contributors`.
    Missing records yield an empty mapping.
    """

    db = patch_db or PatchHistoryDB()
    try:
        rec = db.get(patch_id)
    except Exception:
        rec = None
    if not rec:
        return {}
    try:
        roi_deltas = json.loads(getattr(rec, "roi_deltas", "") or "{}")
    except Exception:
        roi_deltas = {}
    return {
        "roi_before": float(getattr(rec, "roi_before", 0.0)),
        "roi_after": float(getattr(rec, "roi_after", 0.0)),
        "roi_delta": float(
            getattr(rec, "roi_delta", getattr(rec, "roi_after", 0.0) - getattr(rec, "roi_before", 0.0))
        ),
        "roi_deltas": {k: float(v) for k, v in roi_deltas.items()},
    }


# ---------------------------------------------------------------------------
def record_patch_metadata(
    patch_id: int,
    metadata: Mapping[str, Any],
    *,
    patch_db: PatchHistoryDB | None = None,
) -> None:
    """Persist arbitrary metadata for ``patch_id`` in ``patch_history``.

    The metadata is serialised to JSON and stored in the ``summary`` field so
    :mod:`patch_provenance` queries can later reconstruct the full context of a
    patch.
    """

    db = patch_db or PatchHistoryDB()
    try:
        db.record_vector_metrics(
            "", [], patch_id=patch_id, contribution=0.0, win=False, regret=False, summary=json.dumps(metadata)
        )
    except Exception:
        logging.getLogger(__name__).exception("failed to record patch metadata")


# ---------------------------------------------------------------------------
def search_patches_by_vector(
    vector_id: str,
    *,
    patch_db: PatchHistoryDB | None = None,
    limit: int | None = None,
    offset: int = 0,
    index_hint: str | None = None,
) -> List[Dict[str, Any]]:
    """Return patches influenced by ``vector_id`` ordered by influence.

    ``limit`` and ``offset`` provide pagination while ``index_hint`` forces a
    specific SQLite index when querying ``patch_ancestry``.
    """

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_vector(
        vector_id, limit=limit, offset=offset, index_hint=index_hint
    )
    return [
        {
            "patch_id": pid,
            "influence": infl,
            "filename": filename,
            "description": desc,
        }
        for pid, infl, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_hash(
    code_hash: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches matching ``code_hash``."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_hash(code_hash)
    return [
        {"patch_id": pid, "filename": filename, "description": desc}
        for pid, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_license(
    license: str,
    *,
    license_fingerprint: str | None = None,
    patch_db: PatchHistoryDB | None = None,
) -> List[Dict[str, Any]]:
    """Return patches matching ``license`` and optional fingerprint."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_provenance(
        license=license, license_fingerprint=license_fingerprint
    )
    return [
        {"patch_id": pid, "filename": filename, "description": desc}
        for pid, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_license_fingerprint(
    fingerprint: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches matching ``fingerprint``."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_provenance(license_fingerprint=fingerprint)
    return [
        {"patch_id": pid, "filename": filename, "description": desc}
        for pid, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_semantic_alert(
    alert: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches matching ``alert`` semantic alert."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_provenance(semantic_alert=alert)
    return [
        {"patch_id": pid, "filename": filename, "description": desc}
        for pid, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def search_patches(
    license: str | None = None,
    semantic_alert: str | None = None,
    license_fingerprint: str | None = None,
    *,
    patch_db: PatchHistoryDB | None = None,
) -> List[Dict[str, Any]]:
    """Return patches filtered by license, fingerprint and/or semantic alert."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_provenance(
        license=license,
        semantic_alert=semantic_alert,
        license_fingerprint=license_fingerprint,
    )
    return [
        {"patch_id": pid, "filename": filename, "description": desc}
        for pid, filename, desc in rows
    ]


# ---------------------------------------------------------------------------
def build_chain(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return ancestry chain for ``patch_id`` including vector provenance."""

    db = patch_db or PatchHistoryDB()
    chain: List[Dict[str, Any]] = []
    for pid, rec in db.get_ancestry_chain(patch_id):
        chain.append(
            {
                "patch_id": pid,
                "filename": rec.filename,
                "parent_patch_id": rec.parent_patch_id,
                "roi_before": rec.roi_before,
                "roi_after": rec.roi_after,
                "roi_delta": rec.roi_delta,
                "vectors": get_patch_provenance(pid, patch_db=db),
            }
        )
    return chain


__all__ = [
    "get_patch_provenance",
    "get_roi_history",
    "search_patches_by_vector",
    "search_patches_by_hash",
    "search_patches_by_license",
    "search_patches_by_license_fingerprint",
    "search_patches_by_semantic_alert",
    "search_patches",
    "build_chain",
    "PatchLogger",
    "record_patch_metadata",
]

