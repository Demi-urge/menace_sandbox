from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass, field
from typing import List, Any

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .data_bot import DataBot, MetricsDB
from .capital_management_bot import CapitalManagementBot
from .bot_registry import BotRegistry

registry = BotRegistry()
data_bot = DataBot(start_server=False)

@self_coding_managed(bot_registry=registry, data_bot=data_bot)
@dataclass
class OversightBot:
    """Monitor subordinate bots using collected metrics."""

    level: str
    data_bot: DataBot
    capital_bot: CapitalManagementBot
    subordinates: List[str] = field(default_factory=list)

    def add_subordinate(self, bot: str) -> None:
        self.subordinates.append(bot)

    def monitor(self, limit: int = 100) -> Any:
        """Return aggregated metrics for subordinate bots."""
        data = self.data_bot.db.fetch(limit)
        if pd is None:
            filtered = [row for row in data if row.get("bot") in self.subordinates]
            for row in filtered:
                row["roi"] = self.data_bot.roi(row.get("bot", ""))
            return filtered
        df = data[data["bot"].isin(self.subordinates)].copy()
        df["roi"] = df["bot"].apply(self.data_bot.roi)
        summary = (
            df.groupby("bot")[["cpu", "memory", "response_time", "errors", "revenue", "expense", "roi"]]
            .mean()
            .reset_index()
        )
        return summary


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class L1OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("L1", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class L2OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("L2", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class L3OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("L3", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class M1OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("M1", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class M2OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("M2", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class M3OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("M3", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class H1OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("H1", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class H2OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("H2", data_bot, capital_bot)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class H3OversightBot(OversightBot):
    def __init__(self, data_bot: DataBot | None = None, capital_bot: CapitalManagementBot | None = None) -> None:
        data_bot = data_bot or DataBot(MetricsDB())
        capital_bot = capital_bot or CapitalManagementBot(data_bot=data_bot)
        super().__init__("H3", data_bot, capital_bot)


__all__ = [
    "OversightBot",
    "L1OversightBot",
    "L2OversightBot",
    "L3OversightBot",
    "M1OversightBot",
    "M2OversightBot",
    "M3OversightBot",
    "H1OversightBot",
    "H2OversightBot",
    "H3OversightBot",
]
