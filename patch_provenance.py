from __future__ import annotations

"""Helpers for querying patch vector ancestry information."""

from typing import Any, Dict, List

from code_database import PatchHistoryDB


# ---------------------------------------------------------------------------
def get_patch_provenance(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return vectors influencing ``patch_id`` ordered by contribution.

    Results are read from the ``patch_ancestry`` table and include the
    original rank and contribution score for each vector.
    """

    db = patch_db or PatchHistoryDB()
    with db._connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            "SELECT vector_id, rank, contribution FROM patch_ancestry "
            "WHERE patch_id=? ORDER BY contribution DESC",
            (patch_id,),
        ).fetchall()
    return [
        {
            "vector_id": vid,
            "rank": int(rank),
            "contribution": float(contrib),
        }
        for vid, rank, contrib in rows
    ]


# ---------------------------------------------------------------------------
def search_patches_by_vector(
    vector_id: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return patches influenced by ``vector_id`` ordered by contribution."""

    db = patch_db or PatchHistoryDB()
    like = f"%{vector_id}%"
    with db._connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            "SELECT a.patch_id, a.contribution, h.filename, h.description "
            "FROM patch_ancestry a JOIN patch_history h ON h.id=a.patch_id "
            "WHERE a.vector_id LIKE ? ORDER BY a.contribution DESC",
            (like,),
        ).fetchall()
    return [
        {
            "patch_id": int(pid),
            "contribution": float(contrib),
            "filename": filename,
            "description": desc,
        }
        for pid, contrib, filename, desc in rows
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

