from __future__ import annotations

"""Helpers for reconstructing patch provenance chains."""

from typing import Any, Dict, List

from .code_database import PatchHistoryDB


def build_chain(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    """Return ancestry chain for ``patch_id`` including provenance data."""

    db = patch_db or PatchHistoryDB()
    chain: List[Dict[str, Any]] = []
    current: int | None = patch_id
    while current is not None:
        rec = db.get(current)
        if rec is None:
            break
        prov = db.get_provenance(current)
        chain.append(
            {
                "patch_id": current,
                "filename": rec.filename,
                "parent_patch_id": rec.parent_patch_id,
                "vectors": [
                    {
                        "origin": o,
                        "vector_id": vid,
                        "influence": score,
                        "retrieved_at": ts,
                    }
                    for o, vid, score, ts in prov
                ],
            }
        )
        current = rec.parent_patch_id
    return chain


__all__ = ["build_chain"]

