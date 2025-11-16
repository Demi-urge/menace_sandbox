from __future__ import annotations
"""Real-time alert dispatcher for Security AI events."""

import json
import logging
import smtplib
import ssl
import time
from datetime import datetime
from email.message import EmailMessage
from typing import Any, Dict

from dynamic_path_router import get_project_root, resolve_path

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


CONFIG_PATH = resolve_path("config/alert_settings.json")
LOG_PATH = get_project_root() / "logs" / "alert_failures.log"

logger = logging.getLogger("AlertDispatcher")
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log_failure(message: str) -> None:
    """Write *message* to the failure log."""
    try:
        logger.error(message)
    except Exception:
        # fallback plain file write if logger misconfigured
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.time()}: {message}\n")


def _load_config() -> dict[str, Any]:
    """Load alert configuration from ``CONFIG_PATH``."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        _log_failure(f"alert config not found at {CONFIG_PATH}")
    except json.JSONDecodeError as exc:
        _log_failure(f"invalid alert config: {exc}")
    except Exception as exc:  # pragma: no cover - unexpected errors
        _log_failure(f"failed loading alert config: {exc}")
    return {}


CONFIG: Dict[str, Any] = _load_config()


def send_discord_alert(message: str, webhook_url: str) -> bool:
    """Post an alert *message* to Discord via *webhook_url*."""
    if requests is None:
        _log_failure("requests library missing for Discord alert")
        return False
    payload = {"content": message}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as exc:  # pragma: no cover - network issues
        _log_failure(f"Discord alert failed: {exc}")
        return False


def send_email_alert(
    subject: str,
    message: str,
    recipient_email: str,
    sender_email: str,
    smtp_server: str,
    smtp_port: int,
    login: str,
    password: str,
) -> bool:
    """Send an email alert using SMTP."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content(message)

    context = ssl.create_default_context()
    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=10) as server:
                server.login(login, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls(context=context)
                server.login(login, password)
                server.send_message(msg)
        return True
    except Exception as exc:  # pragma: no cover - network issues
        _log_failure(f"Email alert failed: {exc}")
        return False


def dispatch_alert(
    alert_type: str,
    severity: int,
    message: str,
    context: Dict[str, Any] | None = None,
) -> None:
    """Dispatch an alert to the appropriate channels."""
    cfg = CONFIG
    if not cfg:
        _log_failure("dispatch attempted without valid config")
        return

    context = context or {}
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    formatted = f"[{timestamp}] [{alert_type}] {message}"
    if ctx:
        formatted += f" ({ctx})"

    webhook = cfg.get("discord_webhook")
    if webhook:
        send_discord_alert(formatted, webhook)

    threshold = int(cfg.get("severity_threshold", 4))
    if severity >= threshold:
        email_cfg = cfg.get("email", {})
        if email_cfg:
            send_email_alert(
                f"Security AI Alert: {alert_type}",
                formatted,
                email_cfg.get("recipient", ""),
                email_cfg.get("sender", ""),
                email_cfg.get("smtp_server", ""),
                int(email_cfg.get("smtp_port", 0)),
                email_cfg.get("login", ""),
                email_cfg.get("password", ""),
            )


__all__ = [
    "dispatch_alert",
    "send_discord_alert",
    "send_email_alert",
    "CONFIG_PATH",
    "LOG_PATH",
]
