"""Helpers for sandbox dashboard alignment summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def load_alignment_flag_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load alignment flag records from *path*.

    The JSON Lines file is parsed into a list where each entry corresponds to
    an improvement cycle. Malformed lines are ignored.
    """

    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return records


__all__ = ["load_alignment_flag_records"]
