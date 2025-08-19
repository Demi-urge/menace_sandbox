from __future__ import annotations

"""Helpers for querying patch vector ancestry information."""

from typing import Any, Dict, List
import json

from code_database import PatchHistoryDB


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
            }
        )
    return result


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
    license: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches matching ``license``."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_provenance(license=license)
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
                "vectors": get_patch_provenance(pid, patch_db=db),
            }
        )
    return chain


__all__ = [
    "get_patch_provenance",
    "search_patches_by_vector",
    "search_patches_by_hash",
    "search_patches_by_license",
    "search_patches_by_semantic_alert",
    "search_patches",
    "build_chain",
]

