from __future__ import annotations

"""Weekly report generation utilities for Security AI logs."""

import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
from typing import Any, Dict, List

from dynamic_path_router import resolve_path

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

# ---------------------------------------------------------------------------
# File paths for log inputs and report outputs
LOG_DIR = resolve_path("logs")
AUDIT_LOG = LOG_DIR / "audit_log.jsonl"
VIOLATION_LOG = LOG_DIR / "violation_log.jsonl"
LOCK_HISTORY_LOG = LOG_DIR / "lock_history.jsonl"

REPORT_DIR = resolve_path("reports")

logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    """Return ``datetime`` for *value* if possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception as exc:
            logger.warning("failed to parse ISO timestamp %s", value, exc_info=exc)
        try:
            return datetime.utcfromtimestamp(float(value))
        except Exception as exc:
            logger.warning("failed to parse numeric timestamp %s", value, exc_info=exc)
            return None
    return None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file returning a list of dicts."""
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ---------------------------------------------------------------------------

def generate_weekly_report(start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
    """Return a weekly summary of Security AI activity."""

    now = datetime.utcnow()
    end_dt = datetime.fromisoformat(end_date) if end_date else now
    start_dt = datetime.fromisoformat(start_date) if start_date else end_dt - timedelta(days=7)

    # Load logs
    audit_entries = _load_jsonl(AUDIT_LOG)
    violation_entries = _load_jsonl(VIOLATION_LOG)
    lock_entries = _load_jsonl(LOCK_HISTORY_LOG)

    # Filter by timestamp
    def within(entry: Dict[str, Any]) -> bool:
        ts = _parse_timestamp(entry.get("timestamp"))
        if ts is None:
            return False
        return start_dt <= ts <= end_dt

    audit_entries = [e for e in audit_entries if within(e)]
    violation_entries = [e for e in violation_entries if within(e)]
    lock_entries = [e for e in lock_entries if within(e)]

    # Summary metrics -----------------------------------------------------
    total_actions = len(audit_entries)

    rewards: List[float] = []
    risk_scores: List[float] = []
    domains: List[str] = []

    for entry in audit_entries:
        data = entry.get("data", {}) if isinstance(entry.get("data"), dict) else entry
        # Extract reward value if present
        for key in ("reward", "reward_value", "reward_score"):
            if key in data and isinstance(data[key], (int, float)):
                rewards.append(float(data[key]))
                break
        # Extract risk score if present
        for key in ("risk_score", "final_risk_score", "risk"):
            if key in data and isinstance(data[key], (int, float)):
                risk_scores.append(float(data[key]))
                break
        # Extract accessed domain if present
        for key in ("target_domain", "domain", "url", "host"):
            if key in data:
                domains.append(str(data[key]))
                break

    avg_reward = statistics.mean(rewards) if rewards else 0.0

    risk_distribution: Dict[str, float | None]
    if risk_scores:
        risk_distribution = {
            "min": min(risk_scores),
            "max": max(risk_scores),
            "average": statistics.mean(risk_scores),
        }
    else:
        risk_distribution = {"min": None, "max": None, "average": None}

    top_domains = [
        {"domain": d, "count": c} for d, c in Counter(domains).most_common(5)
    ]

    # Violations grouped by type
    violations_by_type: Dict[str, int] = defaultdict(int)
    for v in violation_entries:
        vtype = str(v.get("violation_type", "unknown"))
        violations_by_type[vtype] += 1

    # Lockdown event summaries
    lockdown_events: List[Dict[str, Any]] = []
    for e in lock_entries:
        ts = _parse_timestamp(e.get("timestamp"))
        lockdown_events.append(
            {
                "timestamp": ts.isoformat() if ts else None,
                "reason": e.get("reason"),
                "severity": e.get("severity"),
            }
        )

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "total_actions_evaluated": total_actions,
        "average_reward": avg_reward,
        "top_domains_accessed": top_domains,
        "risk_score_distribution": risk_distribution,
        "violations_logged": dict(violations_by_type),
        "lockdown_events": lockdown_events,
    }


# ---------------------------------------------------------------------------

def write_report(report_data: Dict[str, Any], output_path: str) -> None:
    """Write *report_data* to JSON and plaintext in *output_path*."""

    output_dir = resolve_path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "weekly_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)

    # Build a simple human-readable summary
    lines = [
        f"Weekly Report {report_data.get('start')} - {report_data.get('end')}",
        f"Total actions evaluated: {report_data.get('total_actions_evaluated')}",
        f"Average reward: {report_data.get('average_reward')}",
        "Top domains accessed:",
    ]
    for item in report_data.get("top_domains_accessed", []):
        lines.append(f"  - {item['domain']}: {item['count']}")
    lines.append("Risk score distribution:")
    dist = report_data.get("risk_score_distribution", {})
    for key in ("min", "max", "average"):
        lines.append(f"  {key}: {dist.get(key)}")
    lines.append("Violations logged:")
    for vtype, count in report_data.get("violations_logged", {}).items():
        lines.append(f"  - {vtype}: {count}")
    lines.append("Lockdown events:")
    for ev in report_data.get("lockdown_events", []):
        lines.append(
            f"  - {ev.get('timestamp')} {ev.get('reason')} (severity {ev.get('severity')})"
        )

    text_path = output_dir / "weekly_report.txt"
    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------

def send_to_discord(report_path: str, webhook_url: str) -> None:
    """Post *report_path* contents to Discord via *webhook_url* if possible."""

    if not webhook_url or requests is None:
        return
    if not os.path.exists(report_path):
        return
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception as exc:
        logger.exception("failed to read report %s", report_path, exc_info=exc)
        return
    try:
        requests.post(webhook_url, json={"content": f"```{content}```"})
    except Exception as exc:
        logger.exception("failed to send report to Discord: %s", exc)


__all__ = [
    "generate_weekly_report",
    "write_report",
    "send_to_discord",
]
