from __future__ import annotations

"""Simple FastAPI receiver for research items forwarded to Stage 3."""

from typing import Any, Dict, List
import json
import os
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)

app = FastAPI(title="Stage 3 Service")

_DATA_FILE = os.getenv("STAGE3_FILE", "stage3_items.jsonl")


@app.post("/stage3")
async def collect_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Persist *items* to ``_DATA_FILE`` and return a confirmation."""
    try:
        with open(_DATA_FILE, "a", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
    except Exception as exc:  # pragma: no cover - file I/O issues
        logger.exception("Failed to store Stage 3 items: %s", exc)
        return {"ok": False}
    return {"ok": True, "received": len(items)}


__all__ = ["app"]
