from __future__ import annotations

"""Helpers for querying patch vector ancestry information."""

from typing import Any, Dict, List

from code_database import PatchHistoryDB


# ---------------------------------------------------------------------------
def get_patch_provenance(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return vectors influencing ``patch_id`` ordered by influence.

    Results are read from the ``patch_ancestry`` table and include the
    origin and influence score for each vector.
    """

    db = patch_db or PatchHistoryDB()
    rows = db.get_ancestry(patch_id)
    return [
        {
            "origin": origin,
            "vector_id": vid,
            "influence": float(infl),
        }
        for origin, vid, infl in rows
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
def build_chain(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return ancestry chain for ``patch_id`` including vector provenance."""

    db = patch_db or PatchHistoryDB()
    chain: List[Dict[str, Any]] = []
    current: int | None = patch_id
    while current is not None:
        rec = db.get(current)
        if rec is None:
            break
        chain.append(
            {
                "patch_id": current,
                "filename": rec.filename,
                "parent_patch_id": rec.parent_patch_id,
                "vectors": get_patch_provenance(current, patch_db=db),
            }
        )
        current = rec.parent_patch_id
    return chain


__all__ = ["get_patch_provenance", "search_patches_by_vector", "build_chain"]

