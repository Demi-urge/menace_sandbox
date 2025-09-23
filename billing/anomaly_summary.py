"""Summarise recent billing anomalies and record insights."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Any

from menace_sanity_layer import list_anomalies
from menace_sandbox.gpt_memory import GPTMemoryManager
from log_tags import FEEDBACK
from metrics_aggregator import MetricsDB


def summarise_anomalies(
    limit: int = 100,
    *,
    memory: GPTMemoryManager | None = None,
    metrics_db: MetricsDB | None = None,
) -> Dict[str, Any]:
    """Generate insights from recent billing anomalies.

    Parameters
    ----------
    limit:
        Maximum number of anomalies to analyse.
    memory:
        Optional :class:`GPTMemoryManager` instance for logging feedback.
    metrics_db:
        Optional :class:`metrics_aggregator.MetricsDB` instance for persisting
        evaluation metrics.

    Returns
    -------
    dict
        Summary containing total anomalies, top offending bots and trending
        anomaly types.
    """

    anomalies = list_anomalies(limit)
    bot_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for entry in anomalies:
        meta = entry.get("metadata") or {}
        bot_id = str(meta.get("bot_id", "unknown"))
        bot_counts[bot_id] += 1
        event_type = entry.get("event_type", "unknown")
        type_counts[event_type] += 1

    top_offenders = bot_counts.most_common()
    trending_issues = type_counts.most_common()

    summary: Dict[str, Any] = {
        "total": len(anomalies),
        "top_offenders": top_offenders,
        "trending_issues": trending_issues,
    }

    # Log insight to GPT memory
    try:
        mem = memory or GPTMemoryManager()
        prompt = "billing.anomaly_summary"
        lines = [
            f"total anomalies: {len(anomalies)}",
            "top offenders: "
            + ", ".join(f"{b}:{c}" for b, c in top_offenders[:5]),
            "trending issues: "
            + ", ".join(f"{t}:{c}" for t, c in trending_issues[:5]),
        ]
        mem.log_interaction(prompt, "\n".join(lines), tags=[FEEDBACK, "billing"])
    except Exception:
        pass

    # Persist metrics
    try:
        db = metrics_db or MetricsDB()
        cycle = "billing.anomaly_summary"
        db.log_eval(cycle, "total_anomalies", float(len(anomalies)))
        for bot, count in top_offenders:
            db.log_eval(cycle, f"bot.{bot}", float(count))
        for typ, count in trending_issues:
            db.log_eval(cycle, f"type.{typ}", float(count))
    except Exception:
        pass

    return summary
