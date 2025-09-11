"""High-level interface for model idea processing."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from pathlib import Path
from typing import Iterable

from .db_router import DBRouter
from .admin_bot_base import AdminBotBase

from .preliminary_research_bot import PreliminaryResearchBot
from .capital_management_bot import CapitalManagementBot
from .database_manager import (
    process_idea,
    update_profitability_threshold,
    DB_PATH,
)


@self_coding_managed
class DatabaseManagementBot(AdminBotBase):
    """Coordinate idea ingestion and profitability thresholds."""

    def __init__(
        self,
        prelim_bot: PreliminaryResearchBot | None = None,
        capital_bot: CapitalManagementBot | None = None,
        db_path: Path = DB_PATH,
        db_router: DBRouter | None = None,
    ) -> None:
        super().__init__(db_router=db_router)
        self.prelim = prelim_bot or PreliminaryResearchBot()
        self.capital = capital_bot or CapitalManagementBot()
        self.db_path = db_path

    # ------------------------------------------------------------------
    def ingest_idea(
        self,
        name: str,
        *,
        tags: Iterable[str] = (),
        source: str = "",
        urls: Iterable[str] = (),
        load: float = 0.0,
        success_rate: float = 1.0,
        deploy_eff: float = 1.0,
        failure_rate: float = 0.0,
    ) -> str:
        """Process a new model suggestion and store the result."""
        self.query(name)
        energy = self.capital.energy_score(
            load=load,
            success_rate=success_rate,
            deploy_eff=deploy_eff,
            failure_rate=failure_rate,
        )
        status = process_idea(
            name,
            tags=tags,
            source=source,
            urls=urls,
            prelim=self.prelim,
            energy_score=energy,
            db_path=self.db_path,
        )
        return status

    # ------------------------------------------------------------------
    def adjust_threshold(self) -> float:
        """Recalculate and apply the profitability threshold."""
        energy = self.capital.energy_score(
            load=0.0, success_rate=1.0, deploy_eff=1.0, failure_rate=0.0
        )
        return update_profitability_threshold(energy, db_path=self.db_path)


__all__ = ["DatabaseManagementBot"]
