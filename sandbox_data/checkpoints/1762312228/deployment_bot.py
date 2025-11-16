"""Deployment Bot for provisioning and deploying bots."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
# flake8: noqa
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore
from typing import Any, Dict, Iterable, List, Optional


def _log_exception(logger: logging.Logger, context: str, exc: Exception) -> None:
    """Helper to log exceptions with context."""
    logger.exception("%s failed: %s", context, exc)

from .unified_event_bus import UnifiedEventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry

# Internal imports
from .bot_database import BotDB, BotRecord
from .task_handoff_bot import WorkflowDB, WorkflowRecord
from .research_aggregator_bot import InfoDB
from .chatgpt_enhancement_bot import EnhancementDB
from .error_bot import ErrorDB
from .db_router import DBRouter
from .db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .code_database import CodeDB, CodeRecord
from .database_manager import update_model, DB_PATH
from .databases import MenaceDB
from .contrarian_db import ContrarianDB
from .governance import evaluate_rules
from .deployment_governance import evaluate_scorecard
from .borderline_bucket import BorderlineBucket
from .rollback_manager import RollbackManager
from .audit_logger import log_event as audit_log_event
from .scope_utils import Scope, build_scope_clause, apply_scope

# ---------------------------------------------------------------------------
# SQLite layer for deployment & error tracking
# ---------------------------------------------------------------------------

class DeploymentDB:
    """SQLite‑backed deployment & error log using :class:`DBRouter`."""

    def __init__(
        self,
        path: str | Path = "deployment.db",
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        router: DBRouter | None = None,
    ) -> None:
        p = Path(path)
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "deployment", str(p), str(p)
        )
        LOCAL_TABLES.update({"deployments", "errors", "bot_trials", "update_history"})
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        conn = self.router.get_connection("deployments")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deployments(
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT,
                status  TEXT,
                ts      TEXT,
                log     TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                deploy_id  INTEGER,
                message    TEXT,
                ts         TEXT,
                source_menace_id TEXT NOT NULL
            )
            """
        )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(errors)").fetchall()]
        if "source_menace_id" not in cols:
            conn.execute(
                "ALTER TABLE errors ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_errors_source_menace_id ON errors(source_menace_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_trials(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER,
                deploy_id INTEGER,
                status TEXT,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS update_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                packages TEXT,
                status TEXT,
                ts TEXT
            )
            """
        )
        conn.commit()

    # ----------------------------- CRUD helpers ----------------------------

    def add(self, name: str, status: str, log: str) -> int:
        conn = self.router.get_connection("deployments")
        cur = conn.execute(
            "INSERT INTO deployments(name, status, ts, log) VALUES (?,?,?,?)",
            (name, status, datetime.utcnow().isoformat(), log),
        )
        conn.commit()
        did = int(cur.lastrowid)
        if self.event_bus:
            try:
                payload = {
                    "id": did,
                    "name": name,
                    "status": status,
                    "log": log,
                }
                self.event_bus.publish("deployments:new", payload)
            except Exception as exc:
                _log_exception(self.logger, "publish deployments:new", exc)
        return did

    def update_status(self, deploy_id: int, status: str) -> None:
        conn = self.router.get_connection("deployments")
        conn.execute(
            "UPDATE deployments SET status = ? WHERE id = ?",
            (status, deploy_id),
        )
        conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish("deployments:update", {"id": deploy_id, "status": status})
            except Exception as exc:
                _log_exception(self.logger, "publish deployments:update", exc)

    def error(self, deploy_id: int, message: str) -> None:
        conn = self.router.get_connection("errors")
        menace_id = self.router.menace_id if self.router else os.getenv("MENACE_ID", "")
        conn.execute(
            "INSERT INTO errors(source_menace_id, deploy_id, message, ts) VALUES (?,?,?,?)",
            (menace_id, deploy_id, message, datetime.utcnow().isoformat()),
        )
        conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "errors:new",
                    {"deploy_id": deploy_id, "message": message},
                )
            except Exception as exc:
                _log_exception(self.logger, "publish errors:new", exc)

    def errors_for(
        self,
        deploy_id: int,
        *,
        source_menace_id: str | None = None,
        scope: str = "local",
    ) -> List[int]:
        """Return IDs of errors logged for ``deploy_id``.

        The ``scope`` parameter controls menace visibility:

        - ``"local"`` – only errors from the current menace
        - ``"global"`` – errors from other menace instances
        - ``"all"`` – no menace filtering

        ``source_menace_id`` overrides the router's menace ID when provided.
        """

        conn = self.router.get_connection("errors")
        menace_id = source_menace_id or (
            self.router.menace_id if self.router else os.getenv("MENACE_ID", "")
        )
        clause, params = build_scope_clause("errors", Scope(scope), menace_id)
        query = apply_scope("SELECT id FROM errors WHERE deploy_id=?", clause)
        rows = conn.execute(query, (deploy_id, *params)).fetchall()
        return [r[0] for r in rows]

    def get(self, deploy_id: int) -> Dict[str, Any]:
        conn = self.router.get_connection("deployments")
        row = conn.execute(
            "SELECT id,name,status,ts,log FROM deployments WHERE id = ?",
            (deploy_id,),
        ).fetchone()
        if not row:
            return {}
        return {
            "id": row[0],
            "name": row[1],
            "status": row[2],
            "timestamp": row[3],
            "log": row[4],
        }

    # Trial helpers ----------------------------------------------------

    def add_trial(self, bot_id: int, deploy_id: int, status: str = "active") -> int:
        conn = self.router.get_connection("bot_trials")
        cur = conn.execute(
            "INSERT INTO bot_trials(bot_id, deploy_id, status, ts) VALUES (?,?,?,?)",
            (bot_id, deploy_id, status, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)

    def update_trial(self, trial_id: int, status: str) -> None:
        conn = self.router.get_connection("bot_trials")
        conn.execute(
            "UPDATE bot_trials SET status=? WHERE id=?",
            (status, trial_id),
        )
        conn.commit()

    def trials(self, status: str = "active") -> List[dict[str, Any]]:
        conn = self.router.get_connection("bot_trials")
        cur = conn.execute(
            "SELECT id, bot_id, deploy_id, status, ts FROM bot_trials WHERE status=?",
            (status,),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]

    # Update history helpers ------------------------------------------

    def add_update(self, packages: Iterable[str], status: str) -> int:
        conn = self.router.get_connection("update_history")
        cur = conn.execute(
            "INSERT INTO update_history(packages, status, ts) VALUES (?,?,?)",
            (";".join(packages), status, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class DeploymentSpec:
    """Bare‑bones description of a deployment environment."""

    name: str
    resources: Dict[str, Any]  # e.g. {"MyBot": {"cpu":2,"memory":4096}}
    env: Dict[str, str]        # Environment vars

# ---------------------------------------------------------------------------
# Main automation class
# ---------------------------------------------------------------------------

@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class DeploymentBot:
    """Automates build → test → deploy and updates all registry tables."""

    def __init__(
        self,
        db: DeploymentDB | None = None,
        bot_db: BotDB | None = None,
        workflow_db: WorkflowDB | None = None,
        info_db: InfoDB | None = None,
        enh_db: EnhancementDB | None = None,
        code_db: CodeDB | None = None,
        error_db: ErrorDB | None = None,
        menace_db: MenaceDB | None = None,
        contrarian_db: ContrarianDB | None = None,
        db_router: DBRouter | None = None,
        *,
        event_bus: UnifiedEventBus | None = None,
        memory_mgr: MenaceMemoryManager | None = None,
    ) -> None:
        # Databases
        self.db = db or DeploymentDB(event_bus=event_bus)
        self.bot_db = bot_db or BotDB()
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        self.info_db = info_db or InfoDB()
        self.enh_db = enh_db or EnhancementDB()
        self.code_db = code_db or CodeDB()
        self.error_db = error_db or ErrorDB()
        self.menace_db = menace_db
        self.contrarian_db = contrarian_db
        self.db_router = db_router or DBRouter()
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.last_deployment_event: object | None = None
        self.last_memory_entry: MemoryEntry | None = None

        self.rollback_manager = RollbackManager()
        self.borderline_bucket = BorderlineBucket()

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DeploymentBot")
        if self.event_bus:
            try:
                self.event_bus.subscribe("deployments:new", self._on_deployment_event)
            except Exception as exc:
                _log_exception(self.logger, "subscribe deployments:new", exc)
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                _log_exception(self.logger, "subscribe memory manager", exc)

    def errors_for(self, deploy_id: int, *, scope: str = "local") -> List[int]:
        """Delegate to :class:`DeploymentDB.errors_for` with ``scope``."""
        return self.db.errors_for(deploy_id, scope=scope)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_tasks(names: List[str], min_size: int = 3) -> List[List[str]]:
        """Generate overlapping task‑chunks for granular workflows."""
        chunks = [names]
        size = len(names)
        while size > min_size:
            size = max(min_size, size // 2)
            for i in range(0, len(names), size):
                chunk = names[i : i + size]
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Code template extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_code(text: str, *, file_name: str = "<unknown>") -> List[str]:
        """Segment Python file into module + top-level defs/classes for template DB."""
        import ast
        import warnings

        segments = [text]
        try:
            tree = ast.parse(text)
            lines = text.splitlines()
            for node in tree.body:
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    segments.append(seg)
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger("DeploymentBot").exception(
                "AST parsing failed for %s: %s", file_name, exc
            )
            warnings.warn(
                f"failed to parse code from {file_name}: {exc}", RuntimeWarning
            )
            raise ValueError(f"AST parsing failed for {file_name}") from exc
        return segments

    # ------------------------------------------------------------------
    # Workflow & template registration
    # ------------------------------------------------------------------

    def _record_workflows(
        self,
        plan: Iterable[str],
        model_id: int,
        bots: Iterable[str],
        enhancements: Iterable[int],
    ) -> List[int]:
        tasks = list(plan)
        if not tasks:
            return []

        existing = {tuple(rec.workflow): rec.wid for rec in self.workflow_db.fetch()}
        ids: List[int] = []

        for chunk in self._split_tasks(tasks):
            key = tuple(chunk)
            wid = existing.get(key)
            if wid is None:
                rec = WorkflowRecord(
                    workflow=chunk,
                    title=" ".join(chunk[:3]),
                    description="generated from deployment",
                    task_sequence=chunk,
                )
                wid = self.workflow_db.add(rec)
                if wid is None:
                    self.logger.warning("duplicate workflow ignored for %s", key)
                    continue
                existing[key] = wid
            ids.append(wid)

            # Link info → workflow
            try:
                for item in self.info_db.items_for_model(model_id):
                    self.info_db.link_workflow(item.item_id, wid)
            except Exception as exc:
                _log_exception(self.logger, "link info to workflow", exc)

            # Link enhancements → workflow
            for enh in enhancements:
                try:
                    self.enh_db.link_workflow(enh, wid)
                except Exception as exc:
                    _log_exception(self.logger, "link enhancement to workflow", exc)

        # Update model table with primary workflow
        try:
            if ids:
                update_model(model_id, workflow_id=ids[0], db_path=DB_PATH)
        except Exception as exc:
            _log_exception(self.logger, "update model workflow", exc)
        return ids

    # ------------------------------------------------------------------

    def _record_code_templates(
        self,
        bot_map: Dict[str, str],  # bot name → bot_id
        enhancements: Iterable[int],
        errors: Iterable[int],
    ) -> None:
        """Extract & de‑duplicate code segments, then link to registry tables."""
        from difflib import SequenceMatcher

        existing = self.code_db.fetch_all(scope="all")

        for bot_name, bot_id in bot_map.items():
            path = resolve_path(f"{bot_name}.py")
            if not path.exists():
                continue
            text = path.read_text()
            try:
                segments = self._split_code(text, file_name=str(path))
            except Exception as exc:
                _log_exception(self.logger, f"parse code segments from {path}", exc)
                continue
            for segment in segments:
                stripped = segment.lstrip()
                if stripped.startswith("def "):
                    template_type = "function"
                elif stripped.startswith("class "):
                    template_type = "class"
                else:
                    template_type = "module"

                summary = stripped.splitlines()[0][:100] if stripped else ""
                complexity = min(len(segment.splitlines()) / 50.0, 1.0)

                # Deduplication via fuzzy matching
                found_id: int | None = None
                for row in existing:
                    ratio = SequenceMatcher(None, row["code"], segment).ratio()
                    if ratio >= 0.95:
                        found_id = row["id"]
                        # If our segment is *smaller* (likely cleaned‑up), replace code & bump version
                        if len(segment) < len(row["code"]):
                            try:
                                v = float(row.get("version", "1.0"))
                                new_ver = f"{v + 0.1:.1f}"
                            except Exception:
                                new_ver = "1.1"
                            self.code_db.update(found_id, code=segment, version=new_ver)
                            row["code"] = segment  # keep in-memory copy fresh
                        break

                if found_id is None:
                    rec = CodeRecord(
                        code=segment,
                        template_type=template_type,
                        language="python",
                        version="1.0",
                        complexity_score=complexity,
                        summary=summary,
                    )
                    found_id = self.code_db.add(rec)
                    existing.append({"id": found_id, "code": segment, "version": "1.0"})

                # Link tables
                self.code_db.link_bot(found_id, bot_id)
                for enh in enhancements:
                    self.code_db.link_enhancement(found_id, enh)
                for err in errors:
                    self.code_db.link_error(found_id, err)
                    try:
                        self.error_db.link_code(err, found_id)
                    except Exception as exc:
                        _log_exception(self.logger, "link code to error", exc)

    def _log_error(
        self,
        message: str,
        bots: Iterable[str],
        model_id: Optional[int],
    ) -> int:
        """Register deployment error and link to bots, model and code."""
        err_id = self.error_db.add_error(message, type_="deployment")
        if model_id is not None:
            try:
                self.error_db.link_model(err_id, model_id)
            except Exception as exc:
                _log_exception(self.logger, "link error to model", exc)
        for name in bots:
            try:
                rec = self.bot_db.find_by_name(name)
            except Exception as exc:
                _log_exception(self.logger, f"find bot {name}", exc)
                rec = None
            if not rec:
                continue
            bid = rec["id"]
            try:
                self.error_db.link_bot(err_id, bid)
            except Exception as exc:
                _log_exception(self.logger, "link error to bot", exc)
            try:
                for cid in self.code_db.codes_for_bot(bid):
                    self.code_db.link_error(cid, err_id)
                    self.error_db.link_code(err_id, cid)
            except Exception as exc:
                _log_exception(self.logger, "link error to code", exc)
        return err_id

    # ------------------------------------------------------------------
    # Build / test / deploy orchestration
    # ------------------------------------------------------------------

    def prepare_environment(self, spec: DeploymentSpec) -> bool:
        self.logger.info("Preparing environment for %s", spec.name)
        tf_dir = os.getenv("TERRAFORM_DIR")
        if tf_dir and os.path.isdir(tf_dir):
            try:
                subprocess.run(["terraform", "init"], cwd=tf_dir, check=True)
                subprocess.run(["terraform", "apply", "-auto-approve"], cwd=tf_dir, check=True)
            except subprocess.CalledProcessError as exc:
                self.logger.error("Terraform apply failed: %s", exc)
                return False
        return True

    def build_containers(
        self, bots: List[str], *, model_id: Optional[int] = None
    ) -> tuple[List[str], List[int]]:
        images: List[str] = []
        err_ids: List[int] = []
        for bot in bots:
            img = f"{bot}:latest"
            try:
                subprocess.run(["docker", "build", "-t", img, "."], check=True)
                images.append(img)
                self.logger.info("Built image %s", img)
            except subprocess.CalledProcessError as exc:
                self.logger.error("Failed building %s: %s", bot, exc)
                err_ids.append(
                    self._log_error(
                        f"docker build failed for {bot}", [bot], model_id
                    )
                )
        return images, err_ids

    def run_tests(
        self, *, bots: Iterable[str] | None = None, model_id: Optional[int] = None
    ) -> tuple[bool, List[int]]:
        try:
            subprocess.run(["pytest", "-q"], check=True)
            return True, []
        except subprocess.CalledProcessError as exc:
            self.logger.error("Tests failed: %s", exc)
            err_id = self._log_error("tests failed", list(bots or []), model_id)
            return False, [err_id]

    def auto_update_nodes(self, nodes: Iterable[str], branch: str = "main") -> None:
        """Pull latest code and run tests on each node before restarting containers."""
        for node in nodes:
            try:
                # Pull the target branch and execute the project's tests remotely
                test_cmd = f"git pull origin {branch} && pytest -q"
                test_proc = subprocess.run([
                    "ssh",
                    node,
                    test_cmd,
                ], check=False)

                ok = test_proc.returncode == 0
                if ok:
                    subprocess.run(
                        [
                            "ssh",
                            node,
                            "docker compose build --pull && docker compose up -d",
                        ],
                        check=False,
                    )
                if self.event_bus:
                    try:
                        status = "success" if ok else "failed"
                        self.event_bus.publish(
                            "nodes:update",
                            {"node": node, "status": status},
                        )
                    except Exception as exc:
                        _log_exception(self.logger, f"publish nodes:update for {node}", exc)
            except Exception as exc:
                self.logger.error("update failed on %s: %s", node, exc)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "nodes:update",
                            {"node": node, "status": "error", "error": str(exc)},
                        )
                    except Exception as exc2:
                        _log_exception(self.logger, f"publish nodes:update error for {node}", exc2)

    # ------------------------------------------------------------------

    def deploy(
        self,
        name: str,
        bots: List[str],
        spec: DeploymentSpec,
        *,
        model_id: Optional[int] = None,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
        plan: Iterable[str] | None = None,
        contrarian_id: Optional[int] = None,
        errors: Iterable[int] | None = None,
        hierarchy_levels: Dict[str, str] | None = None,
        alignment_status: str = "pass",
        scenario_raroi_deltas: Iterable[float] | None = None,
        override_veto: bool = False,
    ) -> int:
        """Main entry point – returns deployment_id."""
        allow_ship, allow_rollback, reasons = evaluate_rules(
            {}, alignment_status, scenario_raroi_deltas or []
        )
        if not allow_ship:
            for msg in reasons:
                self.logger.warning("governance veto: %s", msg)
            return -1

        scorecard = {"alignment_status": alignment_status}
        deltas = list(scenario_raroi_deltas or [])
        if deltas:
            scorecard["raroi"] = deltas[-1]

        eval_res = evaluate_scorecard(scorecard)
        verdict = eval_res.get("decision")
        reason_codes = eval_res.get("reason_codes", [])
        override_allowed = bool(eval_res.get("override_allowed"))
        try:
            payload = {"verdict": verdict, "reason_codes": reason_codes}
            if verdict in {"borderline", "pilot"}:
                payload["downgrade_type"] = verdict
            audit_log_event("deployment_verdict", payload)
        except Exception as exc:
            _log_exception(self.logger, "audit log", exc)
        self.logger.info(
            "deployment verdict",
            extra={"verdict": verdict, "reason_codes": ";".join(reason_codes)},
        )

        env_override = os.getenv("DEPLOY_OVERRIDE", "").lower() in {"1", "true", "yes"}
        force_override = override_veto or env_override

        if verdict != "promote":
            if override_allowed and force_override:
                self.logger.warning(
                    "governance veto overridden",
                    extra={"verdict": verdict, "reason_codes": ";".join(reason_codes)},
                )
            elif verdict == "demote":
                self.logger.warning("deployment governor demoted workflow")
                try:
                    self.rollback_manager.rollback("latest")
                except Exception as exc:
                    _log_exception(self.logger, "rollback", exc)
                return -1
            elif verdict in {"pilot", "micro_pilot"}:
                self.logger.info("deployment governor requested micro-pilot")
                try:
                    self.borderline_bucket.enqueue(
                        name, scorecard.get("raroi", 0.0), scorecard.get("confidence", 0.0)
                    )
                except Exception as exc:
                    _log_exception(self.logger, "micro-pilot enqueue", exc)
                return -1
            else:
                self.logger.warning("deployment halted: %s", verdict)
                return -1
        if self.db_router:
            try:
                _ = self.db_router.query_all(name)
            except Exception as exc:
                _log_exception(self.logger, "db_router query_all", exc)
        log_blob = json.dumps({"bots": bots, "spec": spec.__dict__})
        deploy_id = self.db.add(name, "started", log_blob)

        contrarian_exists = False
        if contrarian_id is not None and self.contrarian_db:
            try:
                record = self.contrarian_db.get(contrarian_id)
                if record and record.timestamp_last_evaluated:
                    contrarian_exists = True
                    self.contrarian_db.update_timestamp(contrarian_id)
            except Exception as exc:
                _log_exception(self.logger, "contrarian lookup/update", exc)

        err_ids = list(errors or [])

        wf_ids = list(workflows or [])

        success = self.prepare_environment(spec)
        images, build_errs = self.build_containers(bots, model_id=model_id)
        err_ids.extend(build_errs)

        test_ok, test_errs = self.run_tests(bots=bots, model_id=model_id)
        err_ids.extend(test_errs)
        success = success and test_ok

        self.db.update_status(deploy_id, "success" if success else "failed")
        if not success:
            for wid in wf_ids:
                try:
                    self.workflow_db.update_status(wid, "failed")
                except Exception as exc:
                    _log_exception(self.logger, "workflow status update", exc)
            self.db.error(deploy_id, "deployment failed")
            err_ids.append(self._log_error("deployment failed", bots, model_id))
            return deploy_id

        # Post‑deploy bookkeeping
        try:
            wf_ids = list(workflows or [])
            enh_ids = list(enhancements or [])

            if plan and model_id is not None:
                wf_ids = self._record_workflows(plan, model_id, bots, enh_ids)

            for wid in wf_ids:
                try:
                    self.workflow_db.update_status(wid, "active")
                except Exception as exc:
                    _log_exception(self.logger, "workflow status update", exc)

            bot_map = self._update_bot_records(
                bots,
                model_id=model_id,
                workflows=wf_ids,
                enhancements=enh_ids,
                resources=spec.resources,
                levels=hierarchy_levels,
                errors=err_ids,
            )

            self._record_code_templates(bot_map, enhancements=enh_ids, errors=err_ids)

            err_ids.extend(self.db.errors_for(deploy_id))

            if model_id is not None:
                try:
                    update_model(model_id, current_status="active", db_path=DB_PATH)
                except Exception as exc:
                    _log_exception(self.logger, "update model current_status", exc)
                    try:
                        update_model(model_id, exploration_status="active", db_path=DB_PATH)
                    except Exception as exc2:
                        _log_exception(self.logger, "update model exploration_status", exc2)
                if self.menace_db:
                    try:
                        self.menace_db.set_model_status(model_id, "active")
                    except Exception as exc:
                        _log_exception(self.logger, "set model status", exc)
            if contrarian_id is not None and self.contrarian_db:
                try:
                    if not contrarian_exists:
                        for wid in wf_ids:
                            self.contrarian_db.link_workflow(contrarian_id, wid)
                        for enh in enh_ids:
                            self.contrarian_db.link_enhancement(contrarian_id, enh)
                        for err in err_ids:
                            self.contrarian_db.link_error(contrarian_id, err)
                    self.contrarian_db.update_status(contrarian_id, "active")
                except Exception as exc:
                    _log_exception(self.logger, "contrarian registry update", exc)
        except Exception:
            self.logger.exception("Failed to update registry tables post‑deploy")
            self.db.error(deploy_id, "registry update failure")
        return deploy_id

    # ------------------------------------------------------------------
    # Bot registry maintenance
    # ------------------------------------------------------------------

    def _update_bot_records(
        self,
        bots: Iterable[str],
        *,
        model_id: Optional[int],
        workflows: Iterable[int],
        enhancements: Iterable[int],
        resources: Dict[str, Any],
        levels: Dict[str, str] | None = None,
        errors: Iterable[int] | None = None,
    ) -> Dict[str, str]:
        """Ensure BotDB has an entry per bot & all links are established."""
        from difflib import SequenceMatcher

        bot_map: Dict[str, str] = {}
        existing_entries = self.bot_db.fetch_all(scope="all")

        hierarchy_chain = {
            "L1": "L2",
            "L2": "L3",
            "L3": "M1",
            "M1": "M2",
            "M2": "M3",
            "M3": "H1",
            "H1": "H2",
            "H2": "H3",
        }
        last_by_level: Dict[str, str] = {}

        for name in bots:
            level = (levels or {}).get(name, "")
            parent_level = hierarchy_chain.get(level, "")
            parent_id = last_by_level.get(parent_level, "")
            deps = [parent_id] if parent_id else []

            match = self.bot_db.find_by_name(name)
            bot_id: Optional[str] = None

            if match:
                bot_id = match["id"]
                if parent_id and match.get("parent_id") != parent_id:
                    self.bot_db.update_bot(bot_id, parent_id=parent_id, dependencies=",".join(deps))

                # also check for other similar bots to potentially mark obsolete
                similar_records = [
                    row
                    for row in existing_entries
                    if row["id"] != bot_id
                    and SequenceMatcher(None, name.lower(), row["name"].lower()).ratio() >= 0.9
                ]
                current = self.bot_db.find_by_name(name) or match
                try:
                    cur_time = datetime.fromisoformat(current.get("last_modification_date") or current["creation_date"])
                except Exception:
                    cur_time = datetime.utcnow()
                for sim in similar_records:
                    try:
                        sim_time = datetime.fromisoformat(sim.get("last_modification_date") or sim.get("creation_date"))
                    except Exception:
                        sim_time = datetime.min
                    if sim_time < cur_time:
                        self.bot_db.update_bot(sim["id"], status="obsolete")
                    else:
                        self.bot_db.update_bot(bot_id, status="obsolete")
            else:
                # Fuzzy match to detect potential superseded bots
                similar_records = [
                    row
                    for row in existing_entries
                    if SequenceMatcher(None, name.lower(), row["name"].lower()).ratio() >= 0.9
                ]

                rec = BotRecord(
                    name=name,
                    type_="generic",
                    tasks=[name],
                    parent_id=parent_id,
                    dependencies=deps,
                    resources=resources.get(name, {}),
                    hierarchy_level=level,
                )
                bot_id = self.bot_db.add_bot(rec)
                new_time = datetime.fromisoformat(rec.last_modification_date)

                for sim in similar_records:
                    try:
                        sim_time = datetime.fromisoformat(
                            sim.get("last_modification_date") or sim.get("creation_date")
                        )
                    except Exception:
                        sim_time = datetime.min

                    if sim_time < new_time:
                        self.bot_db.update_bot(sim["id"], status="obsolete")
                    else:
                        self.bot_db.update_bot(bot_id, status="obsolete")

                existing_entries.append(self.bot_db.find_by_name(name))

            if level and bot_id:
                last_by_level[level] = bot_id

            # ----- Linkage -----
            if bot_id:
                bot_map[name] = bot_id
                if model_id is not None:
                    self.bot_db.link_model(bot_id, model_id)
                for wid in workflows:
                    self.bot_db.link_workflow(bot_id, wid)
                    if self.menace_db:
                        try:
                            self.menace_db.link_workflow_bot(wid, int(bot_id))
                        except Exception as exc:
                            _log_exception(self.logger, "link menace workflow bot", exc)
                for enh in enhancements:
                    self.bot_db.link_enhancement(bot_id, enh)
                    try:
                        if self.enh_db:
                            self.enh_db.link_bot(enh, bot_id)
                    except Exception as exc:
                        _log_exception(self.logger, "link enhancement bot", exc)
                for err in errors or []:
                    try:
                        self.error_db.link_bot(err, bot_id)
                    except Exception as exc:
                        _log_exception(self.logger, "link error bot", exc)
        return bot_map

    # ------------------------------------------------------------------

    def rollback(self, deploy_id: int) -> None:
        self.logger.info("Rolling back deployment %s", deploy_id)
        self.db.error(deploy_id, "rollback initiated")

    # ------------------------------------------------------------------
    # Canary & blue/green deployments
    # ------------------------------------------------------------------

    def _metric_value(self, name: str) -> float:
        """Return the current value of a Prometheus metric if available."""
        try:
            from prometheus_client import REGISTRY  # type: ignore
        except Exception:  # pragma: no cover - optional dependency missing
            return 0.0
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == name:
                    try:
                        return float(sample.value)
                    except Exception:
                        return 0.0
        return 0.0

    def _monitor_for_regression(
        self, metric: str, threshold: float, duration: int
    ) -> bool:
        """Check Prometheus metric for regression over a duration."""
        import time

        before = self._metric_value(metric)
        if duration:
            time.sleep(duration)
        after = self._metric_value(metric)
        return (after - before) <= threshold

    def canary_release(
        self,
        name: str,
        bots: list[str],
        spec: DeploymentSpec,
        *,
        percentage: float = 0.1,
        monitor_metric: str = "error_rate",
        regression_threshold: float = 0.0,
        monitor_time: int = 0,
        **kwargs: Any,
    ) -> int:
        """Deploy a subset of bots and rollback if metrics regress."""

        subset_size = max(1, int(len(bots) * percentage))
        subset = bots[:subset_size]
        deploy_id = self.deploy(name, subset, spec, **kwargs)
        ok = self._monitor_for_regression(
            monitor_metric, regression_threshold, monitor_time
        )
        if not ok:
            self.rollback(deploy_id)
        return deploy_id

    def blue_green_deploy(
        self,
        name: str,
        bots: list[str],
        spec: DeploymentSpec,
        *,
        active_group: str = "blue",
        **kwargs: Any,
    ) -> tuple[int, str]:
        """Deploy to the inactive group then switch traffic."""

        new_group = "green" if active_group == "blue" else "blue"
        deploy_id = self.deploy(f"{name}-{new_group}", bots, spec, **kwargs)
        self.logger.info("Switching traffic from %s to %s", active_group, new_group)
        return deploy_id, new_group

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_deployment_event(self, topic: str, payload: object) -> None:
        self.last_deployment_event = payload

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        if "deploy" in (entry.tags or "").lower():
            self.last_memory_entry = entry

# ---------------------------------------------------------------------------
__all__ = [
    "DeploymentDB",
    "DeploymentSpec",
    "DeploymentBot",
    "BotDB",
    "BotRecord",
]