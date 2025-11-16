#!/usr/bin/env python3
"""Central evaluation loop for Security AI.

This script continuously monitors Menace AI actions logged in
/mnt/shared/menace_logs/actions.jsonl, evaluates them using the immutable
kpi_reward_core module, and dispatches rewards via reward_dispatcher.
Audit logs are written locally for every processed action.

The loop is designed to run as a long-lived daemon that enforces safety
without requiring network access or third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
import numpy as np

from .dynamic_path_router import resolve_path
from .kpi_reward_core import compute_reward, explain_reward
from . import reward_dispatcher
from .roi_calculator import ROICalculator
from .governance import evaluate_veto
from .truth_adapter import TruthAdapter
from .logging_utils import log_record
from .deployment_governance import evaluate_workflow
from .borderline_bucket import BorderlineBucket
from .rollback_manager import RollbackManager
from .foresight_tracker import ForesightTracker

try:  # pragma: no cover - metrics optional
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - best effort
    _me = None

if _me is not None:  # pragma: no cover - metrics optional
    try:
        _TA_LOW_CONF_GAUGE = _me.Gauge(
            "truth_adapter_low_confidence",
            "TruthAdapter low confidence flag",
        )
    except Exception:  # pragma: no cover - gauge already registered
        try:
            from prometheus_client import REGISTRY  # type: ignore

            _TA_LOW_CONF_GAUGE = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                "truth_adapter_low_confidence"
            )
        except Exception:
            _TA_LOW_CONF_GAUGE = None
else:  # pragma: no cover - metrics optional
    _TA_LOW_CONF_GAUGE = None

ENABLE_TRUTH_CALIBRATION = (
    os.getenv("ENABLE_TRUTH_CALIBRATION", "1").lower() not in {"0", "false"}
)
TRUTH_ADAPTER = TruthAdapter() if ENABLE_TRUTH_CALIBRATION else None
if TRUTH_ADAPTER is None and _TA_LOW_CONF_GAUGE is not None:  # pragma: no cover - init
    _TA_LOW_CONF_GAUGE.set(0)

load_dotenv()

BORDERLINE_BUCKET = BorderlineBucket()
ROLLBACK_MANAGER = RollbackManager()
FORESIGHT_TRACKER = ForesightTracker()

# Paths for input logs and output records
ACTIONS_FILE = resolve_path(
    os.getenv("ACTIONS_FILE", "menace_logs/actions.jsonl")
)
CURSOR_FILE = resolve_path("last_processed.txt")
AUDIT_DIR = resolve_path("audit_logs")

# Interval between scans (seconds)
SLEEP_INTERVAL = float(os.getenv("SLEEP_INTERVAL", "5"))
FAILURE_THROTTLE_SLEEP = float(os.getenv("FAILURE_THROTTLE_SLEEP", "60"))


def setup_logging() -> None:
    """Configure application logging."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(AUDIT_DIR / "security_ai.log"),
            logging.StreamHandler(),
        ],
    )


def load_cursor() -> int:
    """Return last processed byte offset."""
    try:
        with open(CURSOR_FILE, "r", encoding="utf-8") as fh:
            return int(fh.read().strip() or 0)
    except FileNotFoundError:
        return 0
    except Exception as exc:
        logging.error("Failed to load cursor: %s", exc)
        return 0


def save_cursor(offset: int) -> None:
    """Persist the current byte offset."""
    try:
        with open(CURSOR_FILE, "w", encoding="utf-8") as fh:
            fh.write(str(offset))
    except Exception as exc:
        logging.error("Failed to save cursor: %s", exc)


def flag_risky_behaviour(action: dict[str, Any]) -> List[str]:
    """Return list of flags detected in *action*."""
    flags: List[str] = []
    joined = json.dumps(action).lower()
    if "bypass" in joined:
        flags.append("bypass_attempt")
    if any(term in joined for term in ["darkweb", "malware", "phishing"]):
        flags.append("risky_domain")
    return flags


def _audit_file_for_today() -> Path:
    date_str = time.strftime("%Y-%m-%d")
    return AUDIT_DIR / f"audit_{date_str}.jsonl"


def append_audit(entry: dict[str, Any]) -> None:
    """Append an audit record."""
    try:
        with open(_audit_file_for_today(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logging.error("Failed to write audit entry: %s", exc)


def _dispatch_with_retry(action: dict[str, Any], max_attempts: int = 3) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            reward_dispatcher.dispatch_reward(action)
            return
        except Exception as exc:  # pragma: no cover - rarely exercised in tests
            if attempt == max_attempts:
                raise
            backoff = 2 ** (attempt - 1)
            logging.warning(
                "Dispatch attempt %s failed: %s; retrying in %s s",
                attempt,
                exc,
                backoff,
            )
            time.sleep(backoff)


def process_action(raw_line: str) -> bool:
    """Evaluate a single log entry and optionally compute ROI."""
    try:
        action = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        logging.error("Malformed log entry: %s", exc)
        append_audit({
            "timestamp": int(time.time()),
            "error": "malformed_entry",
            "details": str(exc),
            "raw": raw_line.strip(),
        })
        return False

    flags = flag_risky_behaviour(action)

    roi_score = None
    if "metrics" in action and "roi_profile" in action:
        calc = ROICalculator()
        metrics = {**action["metrics"], **action.get("flags", {})}
        result = calc.calculate(metrics, action["roi_profile"])
        score, vetoed = result.score, result.vetoed
        low_conf = False
        if TRUTH_ADAPTER is not None:
            try:
                arr = np.array(
                    [
                        float(v)
                        for v in metrics.values()
                        if isinstance(v, (int, float))
                    ],
                    dtype=float,
                ).reshape(1, -1)
                realish, low_conf = TRUTH_ADAPTER.predict(arr)
                score = float(realish[0])
                if _TA_LOW_CONF_GAUGE is not None:
                    _TA_LOW_CONF_GAUGE.set(1 if low_conf else 0)
                if low_conf:
                    logging.warning(
                        "TruthAdapter low confidence; consider retraining",
                        extra=log_record(low_confidence=True),
                    )
            except Exception:  # pragma: no cover - prediction best effort
                logging.exception("TruthAdapter predict failed")
        roi_score = score
        if low_conf:
            action.setdefault("flags", {})["low_confidence"] = True
        calc.log_debug(metrics, action["roi_profile"])
        if vetoed:
            append_audit(
                {
                    "timestamp": int(time.time()),
                    "error": "roi_veto",
                    "action": action,
                }
            )
            return False

    # Governance vetoes based on alignment status and scenario RAROI
    vetoes = evaluate_veto(action.get("scorecard"), action.get("alignment_status", "pass"))

    alignment_status = action.get("alignment_status", "pass")
    if flags or ("ship" in vetoes):
        alignment_status = "fail"
    scorecard = dict(action.get("scorecard") or {})
    if roi_score is not None:
        scorecard["raroi"] = roi_score
    scorecard["alignment_status"] = alignment_status
    eval_result = evaluate_workflow(
        scorecard,
        action.get("deployment_policy"),
        foresight_tracker=FORESIGHT_TRACKER,
        workflow_id=action.get("workflow_id"),
        patch=action.get("patch") or [],
        borderline_bucket=BORDERLINE_BUCKET,
    )
    verdict = eval_result.get("verdict")
    if verdict == "promote":
        wf_id = action.get("workflow_id")
        if wf_id is not None:
            try:
                BORDERLINE_BUCKET.promote(str(wf_id))
            except Exception:
                logging.exception("workflow promotion failed")
    elif verdict == "demote":
        try:
            ROLLBACK_MANAGER.rollback("latest")
        except Exception:
            logging.exception("rollback failed")
    elif verdict in {"pilot", "borderline"}:
        wf_id = action.get("workflow_id")
        if wf_id is not None:
            try:
                BORDERLINE_BUCKET.enqueue(
                    str(wf_id),
                    roi_score or 0.0,
                    scorecard.get("confidence") or 0.0,
                    context=action,
                )
            except Exception:
                logging.exception("borderline enqueue failed")

    decision = action.get("decision")
    if decision and decision in vetoes:
        append_audit(
            {
                "timestamp": int(time.time()),
                "error": "governance_veto",
                "decision": decision,
                "action": action,
            }
        )
        return False

    try:
        reward = compute_reward(action)
        explanation = explain_reward(action)
    except Exception as exc:
        logging.error("Reward computation failed: %s", exc)
        append_audit(
            {
                "timestamp": int(time.time()),
                "error": "reward_failure",
                "details": str(exc),
                "action": action,
            }
        )
        return False

    dispatch_error = None
    try:
        _dispatch_with_retry(action)
    except Exception as exc:
        dispatch_error = str(exc)
        logging.error("Reward dispatch failed: %s", exc)

    audit_record = {
        "timestamp": int(time.time()),
        "reward": reward,
        "explanation": explanation,
        "action": action,
        "flags": flags,
    }
    if roi_score is not None:
        audit_record["roi"] = roi_score
        if action.get("flags", {}).get("low_confidence"):
            audit_record["truth_adapter_low_confidence"] = True
    audit_record["workflow_verdict"] = verdict
    audit_record["workflow_reason_codes"] = eval_result.get("reason_codes", [])
    foresight_info = eval_result.get("foresight") or {}
    if foresight_info.get("forecast_id") is not None:
        audit_record["foresight_forecast_id"] = foresight_info.get("forecast_id")
    if foresight_info.get("reason_codes"):
        audit_record["foresight_reason_codes"] = foresight_info.get("reason_codes")
    if dispatch_error:
        audit_record["dispatch_error"] = dispatch_error
    append_audit(audit_record)
    return dispatch_error is None



def main_loop() -> None:
    """Run the continuous evaluation loop."""
    setup_logging()
    logging.info("Starting central evaluation loop")

    offset = load_cursor()
    failures = 0
    while True:
        try:
            if not os.path.exists(ACTIONS_FILE):
                time.sleep(SLEEP_INTERVAL)
                continue

            with open(ACTIONS_FILE, "r", encoding="utf-8") as fh:
                fh.seek(offset)
                for line in fh:
                    current_pos = fh.tell()
                    if not line.strip():
                        offset = current_pos
                        continue
                    success = process_action(line)
                    failures = 0 if success else failures + 1
                    if failures > 20:
                        logging.warning(
                            "Throttling after %s consecutive failures", failures
                        )
                        time.sleep(FAILURE_THROTTLE_SLEEP)
                    offset = current_pos
        except Exception as exc:
            logging.error("Unexpected error reading actions: %s", exc)
        finally:
            save_cursor(offset)
            time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Security AI evaluation loop")
    parser.add_argument("--actions-file", default=ACTIONS_FILE, help="Path to actions log")
    parser.add_argument(
        "--cursor-file", default=str(CURSOR_FILE), help="Path to cursor file"
    )
    parser.add_argument(
        "--audit-dir", default=str(AUDIT_DIR), help="Audit log directory"
    )
    parser.add_argument(
        "--sleep-interval", type=float, default=SLEEP_INTERVAL, help="Sleep seconds"
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Disable TruthAdapter ROI calibration",
    )
    args = parser.parse_args()

    ACTIONS_FILE = args.actions_file
    CURSOR_FILE = resolve_path(args.cursor_file)
    AUDIT_DIR = resolve_path(args.audit_dir)
    SLEEP_INTERVAL = args.sleep_interval

    ENABLE_TRUTH_CALIBRATION = not args.disable_calibration and (
        os.getenv("ENABLE_TRUTH_CALIBRATION", "1").lower() not in {"0", "false"}
    )
    TRUTH_ADAPTER = TruthAdapter() if ENABLE_TRUTH_CALIBRATION else None
    if TRUTH_ADAPTER is None and _TA_LOW_CONF_GAUGE is not None:
        _TA_LOW_CONF_GAUGE.set(0)

    main_loop()
