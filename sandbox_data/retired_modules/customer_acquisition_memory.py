"""Simple in-memory store for customer acquisition interactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class InteractionRecord:
    platform: str
    user_id: str
    pitch_script: str
    language_style: str
    emotional_strategy: str
    conversion: bool
    age: int
    gender: str
    location: str


class CustomerAcquisitionDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records: List[InteractionRecord] = []

    def log(self, record: InteractionRecord) -> None:
        self.records.append(record)

    def list_interactions(self, platform: str) -> List[InteractionRecord]:
        return [r for r in self.records if r.platform == platform]

    def conversion_rates(self) -> Dict[str, float]:
        totals: Dict[str, int] = {}
        converts: Dict[str, int] = {}
        for r in self.records:
            totals[r.pitch_script] = totals.get(r.pitch_script, 0) + 1
            if r.conversion:
                converts[r.pitch_script] = converts.get(r.pitch_script, 0) + 1
        return {p: converts.get(p, 0) / totals[p] for p in totals}

    def best_pitch_by_platform(self, platform: str) -> str | None:
        rates = self.conversion_rates()
        if not rates:
            return None
        return max(rates.items(), key=lambda x: x[1])[0]

__all__ = ["InteractionRecord", "CustomerAcquisitionDB"]
