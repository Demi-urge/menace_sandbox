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
    return [
        {
            "origin": origin,
            "vector_id": vid,
            "influence": float(infl),
            "license": lic,
            "semantic_alerts": json.loads(alerts) if alerts else [],
        }
        for origin, vid, infl, lic, alerts in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_vector(
    vector_id: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches influenced by ``vector_id`` ordered by influence."""

    db = patch_db or PatchHistoryDB()
    rows = db.find_patches_by_vector(vector_id)
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
    "build_chain",
]

