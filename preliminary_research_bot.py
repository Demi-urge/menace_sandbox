"""Preliminary Research Bot for extracting business data for model evaluation."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Dict, Optional
import os
import logging

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore
from collections import Counter
from .prediction_manager_bot import PredictionManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from .db_router import DBRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimal English stop word list to avoid heavy sklearn dependency
ENGLISH_STOP_WORDS = {
    "the",
    "and",
    "is",
    "to",
    "a",
    "of",
    "in",
    "for",
    "on",
    "with",
    "as",
    "at",
    "by",
    "an",
}


@dataclass
class BusinessData:
    """Structured business metrics extracted from the web."""

    model_name: str
    profit_margin: Optional[float] = None
    operational_cost: Optional[float] = None
    market_saturation: Optional[float] = None
    competitors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    from_year: int = 2000
    to_year: int = datetime.utcnow().year
    inconsistencies: List[str] = field(default_factory=list)
    roi_score: Optional[float] = None


@self_coding_managed
class PreliminaryResearchBot:
    """Scrape and analyse business metrics for a model suggestion."""

    prediction_profile = {"scope": ["research"], "risk": ["low"]}

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        prediction_manager: "PredictionManager" | None = None,
        db_steward: "DBRouter" | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.db_steward = db_steward

    def _apply_prediction_bots(self, roi: float, data: BusinessData) -> float:
        """Combine predictions from assigned bots for ROI."""
        if not self.prediction_manager:
            return roi
        score = roi
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                other = pred(
                    [
                        data.profit_margin or 0.0,
                        data.operational_cost or 0.0,
                        data.market_saturation or 0.0,
                    ]
                )
                if isinstance(other, (list, tuple)):
                    other = other[0]
                score = (score + float(other)) / 2.0
            except Exception:
                continue
        return float(score)

    def _fetch(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=10)
        except Exception:
            return ""
        return resp.text if resp.status_code == 200 else ""

    @staticmethod
    def _extract_value(text: str, term: str) -> Optional[float]:
        pattern = rf"{term}[^0-9]*(\d+(?:\.\d+)?)"
        m = re.search(pattern, text, flags=re.I)
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_competitors(text: str) -> List[str]:
        m = re.search(r"competitors?[:\s]+([A-Za-z0-9, ]+)", text, flags=re.I)
        if not m:
            return []
        items = [c.strip() for c in m.group(1).split(',')]
        return [c for c in items if c]

    @staticmethod
    def _keywords(text: str, top_n: int = 5) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        counts = Counter(words)
        return [w for w, _ in counts.most_common(top_n)]

    def process_model(self, name: str, urls: Iterable[str]) -> BusinessData:
        if self.db_steward:
            try:
                matches = self.db_steward.query_all(name).info
            except Exception:
                matches = []
            if matches:
                return BusinessData(model_name=name)

        texts: List[str] = []
        for u in urls:
            texts.append(self._fetch(u))
        soup_texts = [BeautifulSoup(t, "html.parser").get_text(" ") for t in texts]
        combined = " ".join(soup_texts)

        values: Dict[str, List[float]] = {
            "profit_margin": [],
            "operational_cost": [],
            "market_saturation": [],
        }
        competitors: List[str] = []
        for text in soup_texts:
            pm = self._extract_value(text, "profit margin")
            if pm is not None:
                values["profit_margin"].append(pm)
            oc = self._extract_value(text, "operational cost")
            if oc is not None:
                values["operational_cost"].append(oc)
            ms = self._extract_value(text, "market saturation")
            if ms is not None:
                values["market_saturation"].append(ms)
            competitors.extend(self._extract_competitors(text))

        inconsistencies = [k for k, v in values.items() if len(set(v)) > 1]

        def avg(lst: List[float]) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        margin = avg(values["profit_margin"])
        cost = avg(values["operational_cost"])
        roi_score = None
        if margin is not None and cost not in (None, 0):
            roi_score = (margin - cost) / cost

        data = BusinessData(
            model_name=name,
            profit_margin=margin,
            operational_cost=cost,
            market_saturation=avg(values["market_saturation"]),
            competitors=sorted(set(competitors)),
            keywords=self._keywords(combined),
            inconsistencies=inconsistencies,
            roi_score=roi_score,
        )
        if data.roi_score is not None:
            data.roi_score = self._apply_prediction_bots(data.roi_score, data)
        return data


def send_to_evaluation_bot(data: BusinessData) -> None:
    """POST ``data`` to the model evaluation service."""

    if not requests:
        return
    url = os.getenv("EVALUATION_URL", "http://localhost:8000/evaluate")
    try:
        resp = requests.post(url, json=data.__dict__, timeout=5)
        if not resp.ok:
            logger.warning(
                "Evaluation bot returned status %s for %s", resp.status_code, url
            )
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Failed to send data to evaluation bot at %s: %s", url, exc)


__all__ = ["BusinessData", "PreliminaryResearchBot", "send_to_evaluation_bot"]
