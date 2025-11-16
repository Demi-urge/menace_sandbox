"""Normalize scraped business data into NicheCandidate records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Dict


class NicheCandidate:
    """Unified business or product entry."""

    def __init__(
        self,
        platform: str,
        niche: str | None,
        product_name: str,
        price_point: float | None,
        tags: List[str],
        trend_signal: float | None,
        source_url: str | None,
    ) -> None:
        self.platform = platform
        self.niche = niche
        self.product_name = product_name
        self.price_point = price_point
        self.tags = tags
        self.trend_signal = trend_signal
        self.source_url = source_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "niche": self.niche,
            "product_name": self.product_name,
            "price_point": self.price_point,
            "tags": self.tags,
            "trend_signal": self.trend_signal,
            "source_url": self.source_url,
        }


def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_items(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in paths:
        if p.exists():
            items.extend(_load_json(p))
    return items


def normalize(items: Iterable[Dict[str, Any]]) -> List[NicheCandidate]:
    candidates: List[NicheCandidate] = []
    seen: set[str] = set()
    for itm in items:
        platform = str(itm.get("platform", "")).strip()
        product_name = str(itm.get("product_name", "")).strip()
        if not platform or not product_name:
            continue
        key = f"{platform}:{product_name}".lower()
        if key in seen:
            continue
        seen.add(key)
        niche = str(itm.get("niche", "")).strip() or None
        price_raw = itm.get("price_point")
        try:
            price_point = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price_point = None
        tags_raw = itm.get("tags")
        tags = [str(t).strip() for t in tags_raw] if isinstance(tags_raw, list) else []
        signal_raw = itm.get("trend_signal")
        try:
            trend_signal = float(signal_raw) if signal_raw is not None else None
        except (TypeError, ValueError):
            trend_signal = None
        source_url = str(itm.get("source_url", "")).strip() or None
        cand = NicheCandidate(
            platform=platform,
            niche=niche,
            product_name=product_name,
            price_point=price_point,
            tags=tags,
            trend_signal=trend_signal,
            source_url=source_url,
        )
        candidates.append(cand)
    return candidates


def save_candidates(path: Path, candidates: Iterable[NicheCandidate]) -> None:
    data = [c.to_dict() for c in candidates]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(args: Iterable[str]) -> None:
    paths = [Path(a) for a in args]
    items = load_items(paths)
    candidates = normalize(items)
    save_candidates(Path("niche_candidates.json"), candidates)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
