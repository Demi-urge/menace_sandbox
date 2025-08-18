"""Flask service exposing patch provenance queries."""

from __future__ import annotations

from flask import Flask, jsonify, request
import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB
from patch_provenance import get_patch_provenance, search_patches_by_vector


def _rec_to_dict(rec):
    return {
        "filename": rec.filename,
        "description": rec.description,
        "roi_before": rec.roi_before,
        "roi_after": rec.roi_after,
        "errors_before": rec.errors_before,
        "errors_after": rec.errors_after,
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


def create_app(db: PatchHistoryDB | None = None) -> Flask:
    app = Flask(__name__)
    pdb = db or PatchHistoryDB()

    @app.get("/patches")
    def list_patches():
        patches = [
            {"id": pid, "filename": rec.filename, "description": rec.description}
            for pid, rec in pdb.list_patches()
        ]
        return jsonify(patches)

    @app.get("/patches/<int:patch_id>")
    def show_patch(patch_id: int):
        rec = pdb.get(patch_id)
        if rec is None:
            return jsonify({"error": "not found"}), 404
        prov = get_patch_provenance(patch_id, patch_db=pdb)
        return jsonify({"id": patch_id, "record": _rec_to_dict(rec), "provenance": prov})

    @app.get("/search")
    def search():
        term = request.args.get("q", "")
        if not term:
            return jsonify([])
        res = search_patches_by_vector(term, patch_db=pdb)
        if res:
            patches = [
                {
                    "id": r["patch_id"],
                    "filename": r["filename"],
                    "description": r["description"],
                    "contribution": r["contribution"],
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
                for pid, rec in pdb.search_with_ids(term)
            ]
        return jsonify(patches)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run()
