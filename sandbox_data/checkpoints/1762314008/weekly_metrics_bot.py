"""Bot to summarize weekly MetricsDB statistics and post to Discord."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .data_bot import MetricsDB
from .alert_dispatcher import send_discord_alert

registry = BotRegistry()
data_bot = DataBot(start_server=False)

DEFAULT_WEBHOOK = os.getenv(
    "WEEKLY_METRICS_WEBHOOK", "https://discord.com/api/webhooks/PLACEHOLDER"
)


@dataclass
class WeeklyStats:
    start: str
    end: str
    profit: float
    revenue: float
    expense: float
    roi: float
    top_models: List[Tuple[str, float]]
    delta_profit: float
    delta_revenue: float
    delta_expense: float
    delta_roi: float


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class WeeklyMetricsBot:
    """Aggregate weekly financial metrics and send Discord notifications."""

    def __init__(self, db: MetricsDB | None = None, webhook_url: str = DEFAULT_WEBHOOK) -> None:
        self.db = db or MetricsDB()
        self.webhook_url = webhook_url

    # ------------------------------------------------------------------
    def _sum_metrics(self, rows: object) -> Tuple[float, float, Dict[str, float]]:
        if hasattr(rows, "empty"):
            if getattr(rows, "empty", True):
                return 0.0, 0.0, {}
            revenue = float(rows["revenue"].sum())
            expense = float(rows["expense"].sum())
            grouped = rows.groupby("bot")
            profit_by_bot = {
                str(bot): float(group["revenue"].sum() - group["expense"].sum())
                for bot, group in grouped
            }
            return revenue, expense, profit_by_bot
        if not isinstance(rows, list):
            return 0.0, 0.0, {}
        revenue = sum(float(r.get("revenue", 0.0)) for r in rows)
        expense = sum(float(r.get("expense", 0.0)) for r in rows)
        profit_by_bot: Dict[str, float] = {}
        for r in rows:
            bot = str(r.get("bot"))
            prof = float(r.get("revenue", 0.0) - r.get("expense", 0.0))
            profit_by_bot[bot] = profit_by_bot.get(bot, 0.0) + prof
        return revenue, expense, profit_by_bot

    def _stats_for_range(self, start: datetime, end: datetime) -> Tuple[float, float, float, float, Dict[str, float]]:
        rows = self.db.fetch(limit=None, start=start.isoformat(), end=end.isoformat())
        revenue, expense, profit_by_bot = self._sum_metrics(rows)
        profit = revenue - expense
        roi = (profit / expense) if expense else 0.0
        return revenue, expense, profit, roi, profit_by_bot

    def gather_weekly_stats(self, now: datetime | None = None) -> WeeklyStats:
        now = now or datetime.utcnow()
        end = now
        start = end - timedelta(days=7)
        prev_start = start - timedelta(days=7)

        rev_curr, exp_curr, prof_curr, roi_curr, bot_curr = self._stats_for_range(start, end)
        rev_prev, exp_prev, prof_prev, roi_prev, _ = self._stats_for_range(prev_start, start)

        total_profit = prof_curr or 1.0
        top = sorted(bot_curr.items(), key=lambda x: x[1], reverse=True)[:3]
        top_models = [(b, (p / total_profit) * 100.0) for b, p in top]

        return WeeklyStats(
            start=start.isoformat(),
            end=end.isoformat(),
            profit=prof_curr,
            revenue=rev_curr,
            expense=exp_curr,
            roi=roi_curr,
            top_models=top_models,
            delta_profit=prof_curr - prof_prev,
            delta_revenue=rev_curr - rev_prev,
            delta_expense=exp_curr - exp_prev,
            delta_roi=roi_curr - roi_prev,
        )

    # ------------------------------------------------------------------
    def format_message(self, stats: WeeklyStats) -> str:
        lines = [
            f"Weekly Metrics {stats.start} - {stats.end}",
            f"Profit: {stats.profit:.2f} ({stats.delta_profit:+.2f})",
            f"Revenue: {stats.revenue:.2f} ({stats.delta_revenue:+.2f})",
            f"Expenses: {stats.expense:.2f} ({stats.delta_expense:+.2f})",
            f"ROI: {stats.roi:.2f} ({stats.delta_roi:+.2f})",
            "Top models:",
        ]
        for bot, pct in stats.top_models:
            lines.append(f"  - {bot} ({pct:.1f}%)")
        return "\n".join(lines)

    def send_weekly_report(self) -> None:
        stats = self.gather_weekly_stats()
        message = self.format_message(stats)
        send_discord_alert(message, self.webhook_url)


__all__ = ["WeeklyMetricsBot", "WeeklyStats"]