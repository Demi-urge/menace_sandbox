"""Flask service exposing patch provenance queries.

Supports filtering by license, semantic alerts and license fingerprints when
listing patches.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
import sys
import types
import tempfile
from pathlib import Path

sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)

from code_database import PatchHistoryDB, PatchRecord
from patch_provenance import (
    get_patch_provenance,
    search_patches_by_vector,
    build_chain,
    search_patches,
    search_patches_by_license_fingerprint,
)
from analysis.semantic_diff_filter import find_semantic_risks
from patch_suggestion_db import PatchSuggestionDB
from patch_safety import PatchSafety


def _rec_to_dict(rec):
    return {
        "filename": rec.filename,
        "description": rec.description,
        "roi_before": rec.roi_before,
        "roi_after": rec.roi_after,
        "errors_before": rec.errors_before,
        "errors_after": rec.errors_after,
        "tests_failed_before": rec.tests_failed_before,
        "tests_failed_after": rec.tests_failed_after,
        "roi_delta": rec.roi_delta,
        "complexity_before": rec.complexity_before,
        "complexity_after": rec.complexity_after,
        "complexity_delta": rec.complexity_delta,
        "entropy_before": rec.entropy_before,
        "entropy_after": rec.entropy_after,
        "entropy_delta": rec.entropy_delta,
        "predicted_roi": rec.predicted_roi,
        "predicted_errors": rec.predicted_errors,
        "reverted": rec.reverted,
        "trending_topic": rec.trending_topic,
        "ts": rec.ts,
        "code_id": rec.code_id,
        "code_hash": rec.code_hash,
        "source_bot": rec.source_bot,
        "version": rec.version,
        "parent_patch_id": rec.parent_patch_id,
        "reason": rec.reason,
        "trigger": rec.trigger,
    }


def create_app(
    db: PatchHistoryDB | None = None,
    suggestion_db: PatchSuggestionDB | None = None,
    safety: PatchSafety | None = None,
) -> Flask:
    app = Flask(__name__)
    pdb = db or PatchHistoryDB()
    sdb = suggestion_db or PatchSuggestionDB(
        Path(tempfile.gettempdir()) / "suggestions.db"
    )
    safety_checker = safety or PatchSafety()

    @app.get("/patches")
    def list_patches():
        lic = request.args.get("license")
        alert = request.args.get("semantic_alert")
        fp = request.args.get("license_fingerprint")
        if lic or alert or fp:
            if fp and not lic and not alert:
                rows = search_patches_by_license_fingerprint(fp, patch_db=pdb)
            else:
                rows = search_patches(
                    license=lic,
                    semantic_alert=alert,
                    license_fingerprint=fp,
                    patch_db=pdb,
                )
            patches = [
                {"id": r["patch_id"], "filename": r["filename"], "description": r["description"]}
                for r in rows
            ]
        else:
            patches = [
                {
                    "id": pid,
                    "filename": rec.filename,
                    "description": rec.description,
                    "tests_failed_before": rec.tests_failed_before,
                    "tests_failed_after": rec.tests_failed_after,
                }
                for pid, rec in pdb.list_patches()
            ]
        return jsonify(patches)

    @app.post("/patches")
    def accept_patch():
        data = request.get_json(force=True)
        desc = data.get("description", "")
        module = data.get("module") or data.get("filename", "")
        risks = find_semantic_risks(desc.splitlines())
        if risks:
            sdb.log_decision(module, desc, False, risks[0][1])
            return jsonify({"error": risks[0][1]}), 400
        if safety_checker.is_risky({"category": desc, "module": module}):
            sdb.log_decision(module, desc, False, "failure similarity")
            return jsonify({"error": "failure similarity"}), 400
        rec = PatchRecord(
            data.get("filename", ""),
            desc,
            data.get("roi_before", 0.0),
            data.get("roi_after", 0.0),
        )
        pid = pdb.add(rec)
        sdb.log_decision(module, desc, True, "")
        return jsonify({"id": pid}), 201

    @app.get("/patches/<int:patch_id>")
    def show_patch(patch_id: int):
        rec = pdb.get(patch_id)
        if rec is None:
            return jsonify({"error": "not found"}), 404
        prov = get_patch_provenance(patch_id, patch_db=pdb)
        chain = build_chain(patch_id, patch_db=pdb)
        return jsonify(
            {
                "id": patch_id,
                "record": _rec_to_dict(rec),
                "provenance": prov,
                "chain": chain,
            }
        )

    @app.get("/vectors/<vector_id>")
    def by_vector(vector_id: str):
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", type=int, default=0)
        index_hint = request.args.get("index_hint")
        res = search_patches_by_vector(
            vector_id,
            patch_db=pdb,
            limit=limit,
            offset=offset,
            index_hint=index_hint,
        )
        patches = [
            {
                "id": r["patch_id"],
                "filename": r["filename"],
                "description": r["description"],
                "influence": r["influence"],
            }
            for r in res
        ]
        return jsonify(patches)

    @app.get("/search")
    def search():
        term = request.args.get("q", "")
        if not term:
            return jsonify([])
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", type=int, default=0)
        index_hint = request.args.get("index_hint")
        res = search_patches_by_vector(
            term,
            patch_db=pdb,
            limit=limit,
            offset=offset,
            index_hint=index_hint,
        )
        if res:
            patches = [
                {
                    "id": r["patch_id"],
                    "filename": r["filename"],
                    "description": r["description"],
                    "influence": r["influence"],
                }
                for r in res
            ]
        else:
            patches = [
                {
                    "id": pid,
                    "filename": rec.filename,
                    "description": rec.description,
                }
                for pid, rec in pdb.search_with_ids(
                    term, limit=limit, offset=offset, index_hint=index_hint
                )
            ]
        return jsonify(patches)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run()

