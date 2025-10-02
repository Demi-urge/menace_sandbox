"""Centralised rollback coordination service."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional
import os
import json
import base64
import logging

# ``rollback_manager`` needs to function whether it is imported as part of the
# ``menace_sandbox`` package or executed from a plain source checkout.  Relative
# imports break in the latter scenario (common during environment bootstrap on
# Windows) so we provide explicit fallbacks to absolute imports.
try:  # pragma: no cover - import path handling
    from .db_router import GLOBAL_ROUTER, init_db_router
except ImportError:  # pragma: no cover - script execution fallback
    from db_router import GLOBAL_ROUTER, init_db_router  # type: ignore

router = GLOBAL_ROUTER or init_db_router("rollback_manager")

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:  # pragma: no cover - import path handling
    from .audit_trail import AuditTrail
    from .access_control import READ, WRITE, check_permission
    from .unified_event_bus import UnifiedEventBus
    from .governance import evaluate_rules
except ImportError:  # pragma: no cover - script execution fallback
    from audit_trail import AuditTrail  # type: ignore
    from access_control import READ, WRITE, check_permission  # type: ignore
    from unified_event_bus import UnifiedEventBus  # type: ignore
    from governance import evaluate_rules  # type: ignore



@dataclass
class PatchRecord:
    patch_id: str
    node: str
    applied_at: str


@dataclass
class RegionPatchRecord:
    patch_id: str
    node: str
    file: str
    start_line: int
    end_line: int
    applied_at: str




class RollbackManager:
    """Manage applied patches across distributed nodes."""

    def __init__(
        self,
        path: str = "rollback.db",
        *,
        bot_roles: Dict[str, str] | None = None,
        audit_trail_path: str | None = None,
        audit_privkey: bytes | None = None,
        event_bus: UnifiedEventBus | None = None,
    ) -> None:
        self.path = path
        self._ensure()
        self.bot_roles: Dict[str, str] = bot_roles or {}
        log_path = audit_trail_path or os.getenv("AUDIT_LOG_PATH", "audit.log")
        key_b64 = audit_privkey or os.getenv("AUDIT_PRIVKEY")
        # If no key is provided, disable signing with a warning
        if key_b64:
            priv = base64.b64decode(key_b64) if isinstance(key_b64, str) else key_b64
        else:
            logging.getLogger(__name__).warning(
                "AUDIT_PRIVKEY not set; audit trail entries will not be signed"
            )
            priv = None
        self.audit_trail = AuditTrail(log_path, priv)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus

    def _check_permission(self, action: str, requesting_bot: str | None) -> None:
        if not requesting_bot:
            return
        role = self.bot_roles.get(requesting_bot, READ)
        check_permission(role, action)

    def _log_attempt(self, requesting_bot: str | None, action: str, details: dict) -> None:
        bot = requesting_bot or "unknown"
        ts = datetime.utcnow().isoformat()
        try:
            payload = json.dumps(
                {"timestamp": ts, "bot": bot, "action": action, "details": details},
                sort_keys=True,
            )
            self.audit_trail.record(payload)
        except Exception as exc:
            self.logger.exception("audit trail logging failed for %s", action)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "audit:failed", {"action": action, "error": str(exc)}
                    )
                except Exception:
                    self.logger.exception("event bus publish failed")

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        conn = router.get_connection("patches")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS patches (patch_id TEXT, node TEXT, applied_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS healing_actions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "bot TEXT, action TEXT, patch_id TEXT, ts TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS region_patches ("
            "patch_id TEXT, node TEXT, file TEXT, start_line INTEGER, end_line INTEGER, applied_at TEXT)"
        )
        conn.commit()

    # ------------------------------------------------------------------
    def log_healing_action(
        self, bot: str, action: str, patch_id: str | None = None
    ) -> int:
        ts = datetime.utcnow().isoformat()
        conn = router.get_connection("healing_actions")
        cur = conn.execute(
            "INSERT INTO healing_actions (bot, action, patch_id, ts) VALUES (?, ?, ?, ?)",
            (bot, action, patch_id, ts),
        )
        conn.commit()
        cid = cur.lastrowid

        try:
            payload = json.dumps(
                {
                    "timestamp": ts,
                    "bot": bot,
                    "action": action,
                    "patch_id": patch_id,
                    "change_id": cid,
                },
                sort_keys=True,
            )
            self.audit_trail.record(payload)
        except Exception:
            self.logger.exception("audit trail logging failed")
        return int(cid)

    def healing_actions(self) -> list[dict[str, str]]:
        conn = router.get_connection("healing_actions")
        rows = conn.execute(
            "SELECT id, bot, action, patch_id, ts FROM healing_actions"
        ).fetchall()
        return [
            {
                "id": str(r[0]),
                "bot": r[1],
                "action": r[2],
                "patch_id": r[3],
                "ts": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    def register_patch(self, patch_id: str, node: str) -> None:
        ts = datetime.utcnow().isoformat()
        conn = router.get_connection("patches")
        conn.execute(
            "INSERT INTO patches (patch_id, node, applied_at) VALUES (?, ?, ?)",
            (patch_id, node, ts),
        )
        conn.commit()

    def applied_patches(self, node: str | None = None) -> list[PatchRecord]:
        conn = router.get_connection("patches")
        if node:
            rows = conn.execute(
                "SELECT patch_id, node, applied_at FROM patches WHERE node=?",
                (node,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT patch_id, node, applied_at FROM patches"
            ).fetchall()
        return [PatchRecord(*r) for r in rows]

    # ------------------------------------------------------------------
    def register_region_patch(
        self,
        patch_id: str,
        node: str,
        file: str,
        start_line: int,
        end_line: int,
    ) -> None:
        ts = datetime.utcnow().isoformat()
        conn = router.get_connection("patches")
        conn.execute(
            "INSERT INTO region_patches (patch_id, node, file, start_line, end_line, applied_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (patch_id, node, file, start_line, end_line, ts),
        )
        conn.commit()

    def applied_region_patches(
        self,
        file: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[RegionPatchRecord]:
        conn = router.get_connection("patches")
        if file is not None and start_line is not None and end_line is not None:
            rows = conn.execute(
                "SELECT patch_id, node, file, start_line, end_line, applied_at FROM region_patches "
                "WHERE file=? AND start_line=? AND end_line=?",
                (file, start_line, end_line),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT patch_id, node, file, start_line, end_line, applied_at FROM region_patches",
            ).fetchall()
        return [RegionPatchRecord(*r) for r in rows]

    def rollback(
        self,
        patch_id: str,
        *,
        requesting_bot: str | None = None,
        rpc_client: Optional[object] = None,
        endpoints: Optional[Dict[str, str]] = None,
        alignment_status: str = "pass",
        scenario_raroi_deltas: Iterable[float] | None = None,
    ) -> None:
        """Notify nodes to rollback then drop the patch record."""
        allow_ship, allow_rollback, reasons = evaluate_rules(
            {}, alignment_status, scenario_raroi_deltas or []
        )
        if not allow_rollback:
            for msg in reasons:
                self.logger.warning("governance veto: %s", msg)
            return

        try:
            self._check_permission(WRITE, requesting_bot)
        except PermissionError:
            self._log_attempt(requesting_bot, "rollback_denied", {"patch_id": patch_id})
            raise

        conn = router.get_connection("patches")
        rows = conn.execute(
            "SELECT node FROM patches WHERE patch_id=?",
            (patch_id,),
        ).fetchall()
        nodes = [r[0] for r in rows]

        for node in nodes:
            ok = False
            if rpc_client and hasattr(rpc_client, "rollback"):
                try:
                    ok = bool(rpc_client.rollback(node=node, patch_id=patch_id))
                except Exception:
                    self.logger.exception("rpc rollback failed for %s", node)
            elif endpoints and requests and node in endpoints:
                try:
                    url = endpoints[node].rstrip("/") + "/rollback"
                    resp = requests.post(
                        url, json={"patch_id": patch_id, "node": node}, timeout=5
                    )
                    ok = resp.status_code == 200
                    if not ok:
                        self.logger.warning(
                            "rollback on %s returned %s", node, resp.status_code
                        )
                except Exception:
                    self.logger.exception("http rollback failed for %s", node)
            else:
                self.logger.info("rolling back %s on %s", patch_id, node)
                ok = True

            if not ok:
                self.logger.warning("rollback notification failed for %s", node)

        conn = router.get_connection("patches")
        conn.execute("DELETE FROM patches WHERE patch_id=?", (patch_id,))
        conn.commit()

    def rollback_region(
        self,
        file: str,
        start_line: int,
        end_line: int,
        *,
        requesting_bot: str | None = None,
        rpc_client: Optional[object] = None,
        endpoints: Optional[Dict[str, str]] = None,
        alignment_status: str = "pass",
        scenario_raroi_deltas: Iterable[float] | None = None,
    ) -> None:
        allow_ship, allow_rollback, reasons = evaluate_rules(
            {}, alignment_status, scenario_raroi_deltas or []
        )
        if not allow_rollback:
            for msg in reasons:
                self.logger.warning("governance veto: %s", msg)
            return

        try:
            self._check_permission(WRITE, requesting_bot)
        except PermissionError:
            self._log_attempt(
                requesting_bot,
                "rollback_denied",
                {"file": file, "start_line": start_line, "end_line": end_line},
            )
            raise

        conn = router.get_connection("patches")
        rows = conn.execute(
            "SELECT patch_id, node FROM region_patches WHERE file=? AND start_line=? AND end_line=?",
            (file, start_line, end_line),
        ).fetchall()

        for patch_id, node in rows:
            ok = False
            if rpc_client and hasattr(rpc_client, "rollback_region"):
                try:
                    ok = bool(
                        rpc_client.rollback_region(
                            node=node,
                            patch_id=patch_id,
                            file=file,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                except Exception:
                    self.logger.exception("rpc rollback failed for %s", node)
            elif rpc_client and hasattr(rpc_client, "rollback"):
                try:
                    ok = bool(
                        rpc_client.rollback(
                            node=node,
                            patch_id=patch_id,
                            file=file,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                except Exception:
                    self.logger.exception("rpc rollback failed for %s", node)
            elif endpoints and requests and node in endpoints:
                try:
                    url = endpoints[node].rstrip("/") + "/rollback_region"
                    payload = {
                        "patch_id": patch_id,
                        "node": node,
                        "file": file,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                    resp = requests.post(url, json=payload, timeout=5)
                    ok = resp.status_code == 200
                    if not ok:
                        self.logger.warning(
                            "rollback on %s returned %s", node, resp.status_code
                        )
                except Exception:
                    self.logger.exception("http rollback failed for %s", node)
            else:
                self.logger.info(
                    "rolling back lines %s-%s in %s on %s",
                    start_line,
                    end_line,
                    file,
                    node,
                )
                ok = True

            if not ok:
                self.logger.warning("rollback notification failed for %s", node)

        conn.execute(
            "DELETE FROM region_patches WHERE file=? AND start_line=? AND end_line=?",
            (file, start_line, end_line),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def start_rpc_server(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Expose simple HTTP endpoints for patch registration and rollback."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        mgr = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                import json

                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length).decode()) if length else {}
                if self.path == "/register":
                    mgr.register_patch(data.get("patch_id", ""), data.get("node", ""))
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/rollback":
                    mgr.rollback(data.get("patch_id", ""))
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/register_region":
                    mgr.register_region_patch(
                        data.get("patch_id", ""),
                        data.get("node", ""),
                        data.get("file", ""),
                        int(data.get("start_line", 0)),
                        int(data.get("end_line", 0)),
                    )
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/rollback_region":
                    mgr.rollback_region(
                        data.get("file", ""),
                        int(data.get("start_line", 0)),
                        int(data.get("end_line", 0)),
                    )
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer((host, port), Handler)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop_rpc_server(self) -> None:
        """Stop the running RPC server if active."""
        server = getattr(self, "_server", None)
        if server:
            server.shutdown()
        thread = getattr(self, "_thread", None)
        if thread:
            thread.join(timeout=0)


__all__ = ["RollbackManager", "PatchRecord", "RegionPatchRecord"]
