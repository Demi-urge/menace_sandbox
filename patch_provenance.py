from __future__ import annotations

"""Helpers for querying patch vector ancestry information."""

from typing import Any, Dict, List, Mapping
import json
import logging
import sqlite3

try:  # pragma: no cover - allow package or flat imports
    from .code_database import PatchHistoryDB
    from .vector_service import PatchLogger
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import PatchHistoryDB  # type: ignore
    from vector_service import PatchLogger  # type: ignore


class PatchProvenanceService:
    """High level interface for patch provenance and search helpers."""

    def __init__(self, patch_db: PatchHistoryDB | None = None) -> None:
        self.db = patch_db or PatchHistoryDB()
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def get_provenance(self, patch_id: int) -> List[Dict[str, Any]]:
        """Return vectors influencing ``patch_id`` ordered by influence."""

        rows = self.db.get_ancestry(patch_id)
        try:
            rec = self.db.get(patch_id)
        except sqlite3.DatabaseError as exc:  # pragma: no cover - best effort
            self.logger.exception("failed to fetch patch %s", patch_id)
            raise

        roi_before = float(getattr(rec, "roi_before", 0.0)) if rec else 0.0
        roi_after = float(getattr(rec, "roi_after", 0.0)) if rec else 0.0
        roi_delta = float(getattr(rec, "roi_delta", roi_after - roi_before))
        if roi_delta == 0.0:
            roi_delta = roi_after - roi_before

        result: List[Dict[str, Any]] = []
        for row in rows:
            origin, vid, infl, *rest = row
            lic = rest[0] if len(rest) > 0 else None
            fp = rest[1] if len(rest) > 1 else None
            alerts = rest[2] if len(rest) > 2 else None
            try:
                alerts_list = json.loads(alerts) if alerts else []
            except json.JSONDecodeError:
                alerts_list = []
            result.append(
                {
                    "origin": origin,
                    "vector_id": vid,
                    "influence": float(infl),
                    "license": lic,
                    "license_fingerprint": fp,
                    "semantic_alerts": alerts_list,
                    "roi_before": roi_before,
                    "roi_after": roi_after,
                    "roi_delta": roi_delta,
                }
            )
        return result

    # ------------------------------------------------------------------
    def get_roi_history(self, patch_id: int) -> Dict[str, Any]:
        """Return stored ROI information for ``patch_id``."""

        try:
            rec = self.db.get(patch_id)
        except sqlite3.DatabaseError as exc:  # pragma: no cover - best effort
            self.logger.exception("failed to fetch patch %s", patch_id)
            raise
        if not rec:
            return {}
        try:
            roi_deltas = json.loads(getattr(rec, "roi_deltas", "") or "{}")
        except json.JSONDecodeError:
            roi_deltas = {}
        return {
            "roi_before": float(getattr(rec, "roi_before", 0.0)),
            "roi_after": float(getattr(rec, "roi_after", 0.0)),
            "roi_delta": float(
                getattr(
                    rec,
                    "roi_delta",
                    getattr(rec, "roi_after", 0.0)
                    - getattr(rec, "roi_before", 0.0),
                )
            ),
            "roi_deltas": {k: float(v) for k, v in roi_deltas.items()},
        }

    # ------------------------------------------------------------------
    def record_metadata(self, patch_id: int, metadata: Mapping[str, Any]) -> None:
        """Persist arbitrary metadata for ``patch_id`` in ``patch_history``."""

        try:
            self.db.record_vector_metrics(
                "",
                [],
                patch_id=patch_id,
                contribution=0.0,
                win=False,
                regret=False,
                summary=json.dumps(metadata),
            )
        except sqlite3.DatabaseError:  # pragma: no cover - best effort
            self.logger.exception("failed to record patch metadata")

    # ------------------------------------------------------------------
    def get(self, commit: str) -> Dict[str, Any] | None:
        """Return patch metadata for ``commit`` if present."""

        try:
            rows = self.db.search_with_ids(commit)
        except sqlite3.DatabaseError:  # pragma: no cover - best effort
            self.logger.exception("failed to lookup commit %s", commit)
            return None
        for pid, rec in rows:
            try:
                data = json.loads(getattr(rec, "summary", "") or "{}")
            except json.JSONDecodeError:
                continue
            if data.get("commit") == commit:
                data["patch_id"] = pid
                return data
        return None

    # ------------------------------------------------------------------
    def search_by_vector(
        self,
        vector_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        index_hint: str | None = None,
    ) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_vector(
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

    # ------------------------------------------------------------------
    def search_by_hash(self, code_hash: str) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_hash(code_hash)
        return [
            {"patch_id": pid, "filename": filename, "description": desc}
            for pid, filename, desc in rows
        ]

    # ------------------------------------------------------------------
    def search_by_license(
        self,
        license: str,
        *,
        license_fingerprint: str | None = None,
    ) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_provenance(
            license=license, license_fingerprint=license_fingerprint
        )
        return [
            {"patch_id": pid, "filename": filename, "description": desc}
            for pid, filename, desc in rows
        ]

    # ------------------------------------------------------------------
    def search_by_license_fingerprint(
        self, fingerprint: str
    ) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_provenance(license_fingerprint=fingerprint)
        return [
            {"patch_id": pid, "filename": filename, "description": desc}
            for pid, filename, desc in rows
        ]

    # ------------------------------------------------------------------
    def search_by_semantic_alert(self, alert: str) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_provenance(semantic_alert=alert)
        return [
            {"patch_id": pid, "filename": filename, "description": desc}
            for pid, filename, desc in rows
        ]

    # ------------------------------------------------------------------
    def search(
        self,
        license: str | None = None,
        semantic_alert: str | None = None,
        license_fingerprint: str | None = None,
    ) -> List[Dict[str, Any]]:
        rows = self.db.find_patches_by_provenance(
            license=license,
            semantic_alert=semantic_alert,
            license_fingerprint=license_fingerprint,
        )
        return [
            {"patch_id": pid, "filename": filename, "description": desc}
            for pid, filename, desc in rows
        ]

    # ------------------------------------------------------------------
    def build_chain(self, patch_id: int) -> List[Dict[str, Any]]:
        """Return ancestry chain for ``patch_id`` including vector provenance."""

        chain: List[Dict[str, Any]] = []
        for pid, rec in self.db.get_ancestry_chain(patch_id):
            chain.append(
                {
                    "patch_id": pid,
                    "filename": rec.filename,
                    "parent_patch_id": rec.parent_patch_id,
                    "roi_before": rec.roi_before,
                    "roi_after": rec.roi_after,
                    "roi_delta": rec.roi_delta,
                    "vectors": self.get_provenance(pid),
                }
            )
        return chain


# Backwards compatible functional wrappers -----------------------------------


def get_patch_provenance(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).get_provenance(patch_id)


def get_roi_history(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> Dict[str, Any]:
    return PatchProvenanceService(patch_db).get_roi_history(patch_id)


def record_patch_metadata(
    patch_id: int,
    metadata: Mapping[str, Any],
    *,
    patch_db: PatchHistoryDB | None = None,
) -> None:
    PatchProvenanceService(patch_db).record_metadata(patch_id, metadata)


def get_patch_by_commit(
    commit: str, *, patch_db: PatchHistoryDB | None = None
) -> Dict[str, Any] | None:
    return PatchProvenanceService(patch_db).get(commit)


def search_patches_by_vector(
    vector_id: str,
    *,
    patch_db: PatchHistoryDB | None = None,
    limit: int | None = None,
    offset: int = 0,
    index_hint: str | None = None,
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search_by_vector(
        vector_id, limit=limit, offset=offset, index_hint=index_hint
    )


def search_patches_by_hash(
    code_hash: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search_by_hash(code_hash)


def search_patches_by_license(
    license: str,
    *,
    license_fingerprint: str | None = None,
    patch_db: PatchHistoryDB | None = None,
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search_by_license(
        license, license_fingerprint=license_fingerprint
    )


def search_patches_by_license_fingerprint(
    fingerprint: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search_by_license_fingerprint(
        fingerprint
    )


def search_patches_by_semantic_alert(
    alert: str, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search_by_semantic_alert(alert)


def search_patches(
    license: str | None = None,
    semantic_alert: str | None = None,
    license_fingerprint: str | None = None,
    *,
    patch_db: PatchHistoryDB | None = None,
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).search(
        license=license,
        semantic_alert=semantic_alert,
        license_fingerprint=license_fingerprint,
    )


def build_chain(
    patch_id: int, *, patch_db: PatchHistoryDB | None = None
) -> List[Dict[str, Any]]:
    return PatchProvenanceService(patch_db).build_chain(patch_id)


__all__ = [
    "PatchProvenanceService",
    "get_patch_provenance",
    "get_roi_history",
    "search_patches_by_vector",
    "search_patches_by_hash",
    "search_patches_by_license",
    "search_patches_by_license_fingerprint",
    "search_patches_by_semantic_alert",
    "search_patches",
    "get_patch_by_commit",
    "build_chain",
    "PatchLogger",
    "record_patch_metadata",
]

