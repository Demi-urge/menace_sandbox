"""Bot Creation Bot for designing and deploying new bots."""

# flake8: noqa
from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional
from difflib import SequenceMatcher
from collections import deque
import re
import time

from .data_bot import DataBot, MetricsDB
from .bot_planning_bot import BotPlanningBot, PlanningTask
from .bot_development_bot import BotDevelopmentBot, BotSpec
from .bot_testing_bot import BotTestingBot
from .deployment_bot import DeploymentBot, DeploymentSpec
from .code_database import CodeRecord
from .error_bot import ErrorBot
from .scalability_assessment_bot import ScalabilityAssessmentBot
from .self_coding_engine import SelfCodingEngine
from .safety_monitor import SafetyMonitor
from .prediction_manager_bot import PredictionManager
from .learning_engine import LearningEngine
from .evolution_analysis_bot import EvolutionAnalysisBot
from .workflow_evolution_bot import WorkflowEvolutionBot
from .trending_scraper import TrendingScraper
from .admin_bot_base import AdminBotBase
from datetime import datetime
from .database_manager import DB_PATH, update_model
from vector_service.cognition_layer import CognitionLayer
try:
    from vector_service.context_builder import ContextBuilder
except ImportError:  # pragma: no cover - fallback when helper missing
    from vector_service.context_builder import ContextBuilder  # type: ignore
from .roi_tracker import ROITracker
from .menace_sanity_layer import fetch_recent_billing_issues
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import path_for_prompt  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .intent_clusterer import IntentClusterer
    from .universal_retriever import UniversalRetriever
except Exception:  # pragma: no cover - fallback for flat layout
    from intent_clusterer import IntentClusterer  # type: ignore
    from universal_retriever import UniversalRetriever  # type: ignore
from .coding_bot_interface import self_coding_managed


@dataclass
class CreationConfig:
    """Configuration for when to trigger new bot creation."""

    cpu_threshold: float = 75.0
    complexity_threshold: float = 150.0


@self_coding_managed
class BotCreationBot(AdminBotBase):
    """Identify, build, test and deploy new bots asynchronously."""

    prediction_profile = {"scope": ["creation"], "risk": ["medium"]}

    COOLDOWN_PERIOD = 3600  # seconds
    MAX_BOTS_PER_PERIOD = 5

    def __init__(
        self,
        context_builder: ContextBuilder,
        metrics_db: MetricsDB | None = None,
        planner: BotPlanningBot | None = None,
        developer: BotDevelopmentBot | None = None,
        tester: BotTestingBot | None = None,
        deployer: DeploymentBot | None = None,
        error_bot: ErrorBot | None = None,
        scaler: ScalabilityAssessmentBot | None = None,
        config: CreationConfig | None = None,
        prediction_manager: "PredictionManager" | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        db_router: DatabaseRouter | None = None,
        self_coding_engine: "SelfCodingEngine" | None = None,
        learning_engine: LearningEngine | None = None,
        analysis_bot: EvolutionAnalysisBot | None = None,
        safety_monitor: SafetyMonitor | None = None,
        workflow_bot: "WorkflowEvolutionBot" | None = None,
        trending_scraper: "TrendingScraper" | None = None,
        intent_clusterer: IntentClusterer | None = None,
    ) -> None:
        super().__init__(db_router=db_router)
        self.metrics_db = metrics_db or MetricsDB()
        self.planner = planner or BotPlanningBot()
        self.context_builder = context_builder
        self.developer = developer or BotDevelopmentBot(
            context_builder=self.context_builder, db_steward=self.db_router
        )
        self.tester = tester or BotTestingBot()
        self.deployer = deployer or DeploymentBot()
        self.error_bot = error_bot or ErrorBot(context_builder=self.context_builder)
        self.scaler = scaler or ScalabilityAssessmentBot()
        self.config = config or CreationConfig()
        self.prediction_manager = prediction_manager
        from .capital_management_bot import CapitalManagementBot
        self.capital_bot = capital_bot or CapitalManagementBot(data_bot=DataBot(self.metrics_db))
        self.self_coding_engine = self_coding_engine
        self.safety_monitor = safety_monitor
        self.learning_engine = learning_engine
        self.analysis_bot = analysis_bot
        self.workflow_bot = workflow_bot
        self.trending_scraper = trending_scraper
        self.intent_clusterer = intent_clusterer or IntentClusterer(UniversalRetriever())
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BotCreationBot")
        self._creation_times: Deque[float] = deque()
        self.roi_tracker = ROITracker()
        self.cognition_layer = CognitionLayer(
            roi_tracker=self.roi_tracker, context_builder=self.context_builder
        )
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = DataBot(self.metrics_db)

    def prime(self) -> None:
        """Prime the bot for upcoming creation tasks."""
        # Logging shouldn't fail, but handle defensively
        self.logger.info("Priming BotCreationBot")

    def _apply_prediction_bots(self, base: float, feats: Iterable[float]) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(list(feats))
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue
        return float(score / count)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _log(self, level: int, msg: str) -> None:
        """Central logging helper with optional error dispatch."""
        try:
            self.logger.log(level, msg)
        except Exception:
            pass
        if level >= logging.ERROR and self.error_bot:
            try:
                self.error_bot.db.add_error(msg, type_="runtime", description=msg)
            except Exception:
                pass

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Return a safe bot name consisting of letters, numbers or underscores."""
        safe = re.sub(r"\W+", "_", name)
        return safe.strip("_") or "unnamed_bot"

    def _rate_limited(self) -> bool:
        """Check and update rate limit counters."""
        now = time.time()
        while self._creation_times and now - self._creation_times[0] > self.COOLDOWN_PERIOD:
            self._creation_times.popleft()
        limited = len(self._creation_times) >= self.MAX_BOTS_PER_PERIOD
        if not limited:
            self._creation_times.append(now)
        return limited

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_tasks(names: List[str], min_size: int = 3) -> List[List[str]]:
        """Generate overlapping task chunks for workflow creation."""
        chunks = [names]
        size = len(names)
        while size > min_size:
            size = max(min_size, size // 2)
            for i in range(0, len(names), size):
                chunk = names[i : i + size]
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return chunks

    def _create_workflows(
        self,
        tasks: List[str],
        model_id: int,
        enhancements: Iterable[int],
    ) -> List[int]:
        """Create or reuse workflows for the given plan."""
        from .task_handoff_bot import WorkflowRecord
        from .database_manager import update_model, DB_PATH

        wf_db = self.deployer.workflow_db
        ids: List[int] = []
        chunks = self._split_tasks(tasks)
        priority: List[List[str]] = []
        if self.workflow_bot:
            try:
                suggestions = self.workflow_bot.analyse(limit=5)
                for s in suggestions:
                    seq = s.sequence.split("-")
                    if all(t in tasks for t in seq) and seq not in priority:
                        priority.append(seq)
            except Exception as exc:
                self.logger.error("Workflow bot analysis failed: %s", exc)
        if self.trending_scraper:
            try:
                energy = None
                if self.capital_bot:
                    try:
                        energy = self.capital_bot.energy_score(
                            load=0.0,
                            success_rate=1.0,
                            deploy_eff=1.0,
                            failure_rate=0.0,
                        )
                    except Exception:
                        energy = None
                items = self.trending_scraper.scrape_reddit(energy)[:5]
                names = [
                    (i.product_name or "").replace(" ", "_").lower() for i in items
                ]
                for chunk in chunks:
                    if any(t.lower() in names for t in chunk) and chunk not in priority:
                        priority.append(chunk)
            except Exception as exc:
                self.logger.error("Trending scraper failed: %s", exc)
        ordered = priority + [c for c in chunks if c not in priority]

        for chunk in ordered:
            if self.intent_clusterer:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(
                        " ".join(chunk)
                    )
                    paths = [m.path for m in matches if m.path]
                    clusters = [cid for m in matches for cid in m.cluster_ids]
                    if paths:
                        self.logger.info("intent matches for %s: %s", chunk, paths)
                        for p in paths:
                            name = Path(p).stem
                            if name not in chunk:
                                chunk.append(name)
                    if clusters:
                        self.logger.info(
                            "intent clusters for %s: %s", chunk, clusters
                        )
                except Exception as exc:
                    self.logger.error("intent cluster search failed: %s", exc)
            self.query(" ".join(chunk))
            rec = WorkflowRecord(
                workflow=chunk,
                title=" ".join(chunk[:3]),
                description="generated from creation",
                task_sequence=chunk,
                enhancements=[str(e) for e in enhancements],
                status="pending",
            )
            wid = wf_db.add(rec)
            if wid is None:
                self.logger.warning("duplicate workflow ignored for %s", chunk)
                continue
            ids.append(wid)
            if self.safety_monitor:
                try:
                    self.safety_monitor.validate_workflow(wid, wf_db)
                except Exception as exc:
                    self.logger.error("Safety monitor validation failed: %s", exc)

            # Link related tables
            try:
                for item in self.deployer.info_db.items_for_model(model_id):
                    self.deployer.info_db.link_workflow(item.item_id, wid)
            except Exception as exc:
                self.logger.error("Info registry linking failed: %s", exc)
            for enh in enhancements:
                try:
                    self.deployer.enh_db.link_workflow(enh, wid)
                except Exception as exc:
                    self.logger.error("Enhancement registry linking failed: %s", exc)

        try:
            if ids:
                update_model(model_id, workflow_id=ids[0], db_path=DB_PATH)
        except Exception as exc:
            self.logger.error("Model registry update failed: %s", exc)

        return ids

    # ------------------------------------------------------------------
    # Code template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_code(text: str) -> List[str]:
        """Segment a python file into module and top-level definitions."""
        import ast

        segments = [text]
        try:
            tree = ast.parse(text)
            lines = text.splitlines()
            for node in tree.body:
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    segments.append(seg)
        except Exception as exc:
            self.logger.debug("code parsing failed: %s", exc)
        return segments

    def _record_code_templates(
        self,
        file_path: Path,
        bot_id: str,
        enhancements: Iterable[int],
        errors: Iterable[int],
    ) -> None:
        """Store code segments from a file and link to related tables."""

        if not file_path.exists():
            return

        existing = self.deployer.code_db.fetch_all(scope="all")
        text = file_path.read_text()
        for segment in self._split_code(text):
            stripped = segment.lstrip()
            if stripped.startswith("def "):
                template_type = "function"
            elif stripped.startswith("class "):
                template_type = "class"
            else:
                template_type = "module"

            summary = stripped.splitlines()[0][:100] if stripped else ""
            complexity = min(len(segment.splitlines()) / 50.0, 1.0)

            found_id: int | None = None
            for row in existing:
                ratio = SequenceMatcher(None, row["code"], segment).ratio()
                if ratio >= 0.95:
                    found_id = row["id"]
                    if len(segment) < len(row["code"]):
                        try:
                            v = float(row.get("version", "1.0"))
                            new_ver = f"{v + 0.1:.1f}"
                        except Exception:
                            new_ver = "1.1"
                        self.deployer.code_db.update(
                            found_id, code=segment, version=new_ver
                        )
                        row["code"] = segment
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
                found_id = self.deployer.code_db.add(rec)
                existing.append({"id": found_id, "code": segment, "version": "1.0"})

            self.deployer.code_db.link_bot(found_id, bot_id)
            for enh in enhancements:
                self.deployer.code_db.link_enhancement(found_id, enh)
            for err in errors:
                self.deployer.code_db.link_error(found_id, err)
                try:
                    self.error_bot.db.link_code(err, found_id)
                except Exception as exc:
                    self.logger.error("error db linking failed: %s", exc)

    def _link_existing_code_templates(
        self,
        bot_id: str,
        enhancements: Iterable[int],
        errors: Iterable[int],
    ) -> None:
        """Link enhancements and errors to already stored templates."""
        code_ids = self.deployer.code_db.codes_for_bot(bot_id)
        for cid in code_ids:
            for enh in enhancements:
                self.deployer.code_db.link_enhancement(cid, enh)
            for err in errors:
                self.deployer.code_db.link_error(cid, err)
                try:
                    self.error_bot.db.link_code(err, cid)
                except Exception as exc:
                    self.logger.error("error db linking failed: %s", exc)

    def needs_new_bot(self) -> bool:
        """Check metrics to decide if more bots are required."""
        df = self.metrics_db.fetch(20)
        if df.empty:
            return False
        cpu_avg = float(df["cpu"].mean())
        complexity = DataBot.complexity_score(df)
        base = 1.0 if (
            cpu_avg > self.config.cpu_threshold
            or complexity > self.config.complexity_threshold
        ) else 0.0
        score = self._apply_prediction_bots(base, [cpu_avg, complexity])
        return score > 0.5

    async def _develop_test(self, spec: BotSpec) -> Path:
        safe_name = self._sanitize_name(spec.name)
        spec.name = safe_name
        _ctx, session_id = self.cognition_layer.query(f"build bot {spec.name}")
        try:
            file_path = self.developer.build_bot(
                spec, context_builder=self.developer.context_builder
            )
            self.cognition_layer.record_patch_outcome(session_id, True, contribution=1.0)
        except Exception as exc:
            self.cognition_layer.record_patch_outcome(session_id, False, contribution=0.0)
            self._log(logging.ERROR, f"bot generation failed: {exc}")
            raise
        module_name = spec.name
        _ctx, session_id = self.cognition_layer.query(f"run tests for {module_name}")
        try:
            self.tester.run_unit_tests([module_name])
            self.cognition_layer.record_patch_outcome(session_id, True, contribution=1.0)
        except Exception as exc:
            self.cognition_layer.record_patch_outcome(session_id, False, contribution=0.0)
            raise
        if self.self_coding_engine:
            try:
                prompt_module = path_for_prompt(str(file_path))
                _ctx, session_id = self.cognition_layer.query(
                    f"self coding patch for {prompt_module}"
                )
                billing_instructions = fetch_recent_billing_issues()
                context_meta = (
                    {"billing_instructions": billing_instructions}
                    if billing_instructions
                    else None
                )
                self.self_coding_engine.patch_file(
                    file_path, "helper", context_meta=context_meta
                )
                self.cognition_layer.record_patch_outcome(session_id, True, contribution=1.0)
            except Exception as exc:
                self.cognition_layer.record_patch_outcome(session_id, False, contribution=0.0)
                self._log(logging.ERROR, f"self coding patch failed: {exc}")
        return file_path

    def _hierarchy_levels(self) -> Dict[str, int]:
        bots = self.deployer.bot_db.fetch_all(scope="all")
        order = {
            "L1": 1,
            "L2": 2,
            "L3": 3,
            "M1": 4,
            "M2": 5,
            "M3": 6,
            "H1": 7,
            "H2": 8,
            "H3": 9,
        }
        levels: Dict[str, int] = {}
        id_map = {b["id"]: b for b in bots}
        for b in bots:
            lvl = order.get(b.get("hierarchy_level"), None)
            if lvl is None:
                lvl = 0
                pid = b.get("parent_id")
                while pid:
                    lvl += 1
                    parent = id_map.get(pid)
                    if not parent:
                        break
                    pid = parent.get("parent_id")
            levels[b["id"]] = lvl
        return levels

    def _needs_higher_order(self) -> bool:
        levels = self._hierarchy_levels()
        max_level = max(levels.values() or [0])
        return max_level <= 5 and len(levels) > 2 and self.capital_bot.profit() > 0

    def _update_contrarian_strategy(
        self,
        contrarian_id: int,
        workflows: Iterable[int],
        enhancements: Iterable[int],
        errors: Iterable[int],
    ) -> None:
        """Synchronize contrarian experiment metadata."""
        db = self.deployer.contrarian_db
        if not db:
            return
        try:
            rec = db.get(contrarian_id)
            if rec and rec.timestamp_last_evaluated:
                db.update_timestamp(contrarian_id, datetime.utcnow().isoformat())
            else:
                for wid in workflows:
                    db.link_workflow(contrarian_id, wid)
                for enh in enhancements:
                    db.link_enhancement(contrarian_id, enh)
                for err in errors:
                    db.link_error(contrarian_id, err)
            db.update_status(contrarian_id, "active")
        except Exception as exc:
            self.logger.error("contrarian sync failed: %s", exc)

    async def create_bots(
        self,
        tasks: Iterable[PlanningTask],
        *,
        model_id: Optional[int] = None,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
        contrarian_id: Optional[int] = None,
    ) -> List[int]:
        """Plan, develop, test and deploy bots for tasks."""
        start_time = time.time()
        if self._rate_limited():
            self._log(logging.WARNING, "bot creation rate limit reached")
            self.data_bot.collect(
                bot=self.name,
                response_time=time.time() - start_time,
                errors=0,
                tests_failed=0,
                tests_run=0,
                revenue=0.0,
                expense=0.0,
            )
            return []

        plans = self.planner.plan_bots(list(tasks))
        if self.workflow_bot:
            try:
                suggestions = self.workflow_bot.analyse(limit=5)
                names = {n for s in suggestions for n in s.sequence.split("-")}
                plans.sort(key=lambda p: 0 if p.name in names else 1)
            except Exception as exc:
                self.logger.error("workflow analysis failed: %s", exc)
        if self.trending_scraper:
            try:
                energy = None
                if self.capital_bot:
                    try:
                        energy = self.capital_bot.energy_score(
                            load=0.0,
                            success_rate=1.0,
                            deploy_eff=1.0,
                            failure_rate=0.0,
                        )
                    except Exception:
                        energy = None
                items = self.trending_scraper.scrape_reddit(energy)[:5]
                tnames = [
                    (i.product_name or "").replace(" ", "_").lower() for i in items
                ]
                index_map = {name: idx for idx, name in enumerate(tnames)}
                plans.sort(key=lambda p: index_map.get(p.name.lower(), len(tnames)))
            except Exception as exc:
                self.logger.error("trending sort failed: %s", exc)
        if self.analysis_bot or self.learning_engine:
            try:
                before = self.capital_bot.profit() if self.capital_bot else 0.0
                scored = []
                for p in plans:
                    base = self.analysis_bot.predict("bot_creation", before) if self.analysis_bot else 0.0
                    if self.learning_engine:
                        prob = self.learning_engine.predict_success(
                            p.frequency,
                            p.expected_time,
                            base,
                            float(p.complexity),
                            p.description,
                        )
                        base *= prob or 1.0
                    scored.append((base, p))
                plans = [sp[1] for sp in sorted(scored, key=lambda x: x[0], reverse=True)]
            except Exception as exc:
                self.logger.error("plan scoring failed: %s", exc)
        ids: List[int] = []
        existing = self.deployer.bot_db.fetch_all(scope="all")
        parent_bot_id: Optional[str] = None
        error_msgs = list(self.developer.errors)
        self.developer.errors.clear()
        error_ids: List[int] = []
        for msg in error_msgs:
            eid = self.error_bot.db.add_error(
                msg,
                type_="integration",
                description=msg,
                resolution="fatal",
            )
            if model_id is not None:
                self.error_bot.db.link_model(eid, model_id)
            error_ids.append(eid)

        plan_names = [p.name for p in plans]
        if model_id is not None and not workflows:
            workflows = self._create_workflows(
                plan_names, model_id, list(enhancements or [])
            )
        wf_ids = list(workflows or [])

        level_map = {}
        for plan in plans:
            sanitized = self._sanitize_name(plan.name)
            spec = BotSpec(
                name=sanitized,
                purpose=plan.name,
                functions=["run"],
                level=plan.level,
            )
            level_map[sanitized] = plan.level

            self.query(plan.name)

            blueprint = json.dumps({"tasks": [{"name": sanitized}]})
            report = self.scaler.analyse(blueprint)
            res_cpu = report.tasks[0].cpu if report.tasks else 0.0
            res_mem = report.tasks[0].memory if report.tasks else 0.0
            resources = {sanitized: {"cpu": res_cpu, "memory": res_mem}}

            match_row = None
            for row in existing:
                if (
                    row["name"].lower() == sanitized.lower()
                    or SequenceMatcher(
                        None, sanitized.lower(), row["name"].lower()
                    ).ratio()
                    >= 0.9
                ):
                    match_row = row
                    break

            bot_id: Optional[str] = None
            if match_row:
                bot_id = match_row["id"]
                try:
                    old_res = json.loads(match_row.get("resources", "{}"))
                except Exception:
                    old_res = {}
                better = (
                    bool(old_res)
                    and res_cpu < old_res.get("cpu", float("inf"))
                    and res_mem < old_res.get("memory", float("inf"))
                )
                if better or not old_res:
                    file_path = await self._develop_test(spec)
                    if self.safety_monitor and not self.safety_monitor.validate_bot(spec.name):
                        continue
                    for msg in self.developer.errors:
                        if msg not in error_msgs:
                            error_msgs.append(msg)
                            eid = self.error_bot.db.add_error(
                                msg,
                                type_="integration",
                                description=msg,
                                resolution="fatal",
                            )
                            if model_id is not None:
                                self.error_bot.db.link_model(eid, model_id)
                            error_ids.append(eid)
                    self.developer.errors.clear()
                    dep_spec = DeploymentSpec(
                        name=sanitized, resources=resources, env={}
                    )
                    dep_id = self.deployer.deploy(
                        sanitized,
                        [spec.name],
                        dep_spec,
                        model_id=model_id,
                        workflows=wf_ids,
                        enhancements=enhancements or [],
                        hierarchy_levels=level_map,
                        errors=error_ids,
                        contrarian_id=contrarian_id,
                    )
                    ids.append(dep_id)
                    rec = self.deployer.bot_db.find_by_name(sanitized)
                    bot_id = rec["id"] if rec else bot_id
                    for err in error_ids:
                        self.error_bot.db.link_bot(err, bot_id)
                    for enh in enhancements or []:
                        try:
                            self.deployer.enh_db.link_bot(enh, bot_id)
                        except Exception as exc:
                            self.logger.error("enhancement link failed: %s", exc)
                    if file_path:
                        self._record_code_templates(
                            file_path,
                            bot_id,
                            enhancements or [],
                            error_ids,
                        )
                    try:
                        self.deployer.db.add_trial(bot_id, dep_id, status="active")
                    except Exception as exc:
                        self.logger.error("trial record failed: %s", exc)
                    if model_id is not None:
                        try:
                            update_model(
                                model_id, current_status="active", db_path=DB_PATH
                            )
                        except Exception as exc:
                            self.logger.error("model update failed: %s", exc)
                        if self.deployer.menace_db:
                            try:
                                self.deployer.menace_db.set_model_status(
                                    model_id, "active"
                                )
                            except Exception as db_exc:
                                self.logger.error("menace status update failed: %s", db_exc)
                    if contrarian_id is not None and self.deployer.contrarian_db:
                        try:
                            self.deployer.contrarian_db.update_status(
                                contrarian_id, "active"
                            )
                        except Exception as exc:
                            self.logger.error("contrarian status update failed: %s", exc)
                    if better and old_res:
                            self.deployer.bot_db.update_bot(
                                bot_id, resources=json.dumps(resources[sanitized])
                            )
                else:
                    if model_id is not None:
                        self.deployer.bot_db.link_model(bot_id, model_id)
                    for wid in wf_ids:
                        self.deployer.bot_db.link_workflow(bot_id, wid)
                    for enh in enhancements or []:
                        self.deployer.bot_db.link_enhancement(bot_id, enh)
                        try:
                            self.deployer.enh_db.link_bot(enh, bot_id)
                        except Exception as exc:
                            self.logger.error("enhancement link failed: %s", exc)
                    for err in error_ids:
                        self.error_bot.db.link_bot(err, bot_id)
                    self._link_existing_code_templates(
                        bot_id,
                        enhancements or [],
                        error_ids,
                    )
                    if model_id is not None:
                        try:
                            update_model(
                                model_id, current_status="active", db_path=DB_PATH
                            )
                        except Exception as exc:
                            self.logger.error("model update failed: %s", exc)
                        if self.deployer.menace_db:
                            try:
                                self.deployer.menace_db.set_model_status(
                                    model_id, "active"
                                )
                            except Exception as db_exc:
                                self.logger.error(
                                    "menace status update failed: %s", db_exc
                                )
                    if contrarian_id is not None and self.deployer.contrarian_db:
                        try:
                            self.deployer.contrarian_db.update_status(
                                contrarian_id, "active"
                            )
                        except Exception as exc:
                            self.logger.error("contrarian status update failed: %s", exc)
                    ids.append(0)
            else:
                file_path = await self._develop_test(spec)
                if self.safety_monitor and not self.safety_monitor.validate_bot(spec.name):
                    continue
                for msg in self.developer.errors:
                    if msg not in error_msgs:
                        error_msgs.append(msg)
                        eid = self.error_bot.db.add_error(
                            msg,
                            type_="integration",
                            description=msg,
                            resolution="fatal",
                        )
                        if model_id is not None:
                            self.error_bot.db.link_model(eid, model_id)
                        error_ids.append(eid)
                self.developer.errors.clear()
                dep_spec = DeploymentSpec(name=sanitized, resources=resources, env={})
                dep_id = self.deployer.deploy(
                    sanitized,
                    [spec.name],
                    dep_spec,
                    model_id=model_id,
                    workflows=wf_ids,
                    enhancements=enhancements or [],
                    hierarchy_levels=level_map,
                    errors=error_ids,
                    contrarian_id=contrarian_id,
                )
                ids.append(dep_id)
                rec = self.deployer.bot_db.find_by_name(sanitized)
                if rec:
                    bot_id = rec["id"]
                    for err in error_ids:
                        self.error_bot.db.link_bot(err, bot_id)
                    for enh in enhancements or []:
                        try:
                            self.deployer.enh_db.link_bot(enh, bot_id)
                        except Exception as exc:
                            self.logger.error("enhancement link failed: %s", exc)
                    self._record_code_templates(
                        file_path,
                        bot_id,
                        enhancements or [],
                        error_ids,
                    )
                    try:
                        self.deployer.db.add_trial(bot_id, dep_id, status="active")
                    except Exception as exc:
                        self.logger.error("trial record failed: %s", exc)
                    if model_id is not None:
                        try:
                            update_model(
                                model_id, current_status="active", db_path=DB_PATH
                            )
                        except Exception as exc:
                            self.logger.error("model update failed: %s", exc)
                        if self.deployer.menace_db:
                            try:
                                self.deployer.menace_db.set_model_status(
                                    model_id, "active"
                                )
                            except Exception as db_exc:
                                self.logger.error("menace status update failed: %s", db_exc)
                    if contrarian_id is not None and self.deployer.contrarian_db:
                        try:
                            self.deployer.contrarian_db.update_status(
                                contrarian_id, "active"
                            )
                        except Exception as exc:
                            self.logger.error("contrarian status update failed: %s", exc)

            if bot_id and parent_bot_id:
                self.deployer.bot_db.update_bot(
                    bot_id,
                    parent_id=parent_bot_id,
                    dependencies=parent_bot_id,
                )
            parent_bot_id = bot_id or parent_bot_id
            existing = self.deployer.bot_db.fetch_all(scope="all")

        if self._needs_higher_order():
            name = f"supervisor_{len(existing)+1}"
            sanitized_super = self._sanitize_name(name)
            spec = BotSpec(name=sanitized_super, purpose="manage bots", functions=["run"])
            file_path = await self._develop_test(spec)
            if self.safety_monitor and not self.safety_monitor.validate_bot(spec.name):
                self.data_bot.collect(
                    bot=self.name,
                    response_time=time.time() - start_time,
                    errors=len(error_ids),
                    tests_failed=0,
                    tests_run=0,
                    revenue=0.0,
                    expense=0.0,
                )
                return ids
            for msg in self.developer.errors:
                if msg not in error_msgs:
                    error_msgs.append(msg)
                    eid = self.error_bot.db.add_error(
                        msg,
                        type_="integration",
                        description=msg,
                        resolution="fatal",
                    )
                    if model_id is not None:
                        self.error_bot.db.link_model(eid, model_id)
                    error_ids.append(eid)
            self.developer.errors.clear()
            blueprint = json.dumps({"tasks": [{"name": sanitized_super}]})
            report = self.scaler.analyse(blueprint)
            cpu = report.tasks[0].cpu if report.tasks else 0.0
            mem = report.tasks[0].memory if report.tasks else 0.0
            dep_spec = DeploymentSpec(
                name=sanitized_super,
                resources={sanitized_super: {"cpu": cpu, "memory": mem}},
                env={},
            )
            dep_id = self.deployer.deploy(
                sanitized_super,
                [spec.name],
                dep_spec,
                model_id=model_id,
                workflows=wf_ids,
                enhancements=enhancements or [],
                hierarchy_levels={name: "M3"},
                errors=error_ids,
                contrarian_id=contrarian_id,
            )
            ids.append(dep_id)
            rec = self.deployer.bot_db.find_by_name(sanitized_super)
            if rec:
                bot_id = rec["id"]
                for enh in enhancements or []:
                    try:
                        self.deployer.enh_db.link_bot(enh, bot_id)
                    except Exception as exc:
                        self.logger.error("enhancement link failed: %s", exc)
                self._record_code_templates(
                    file_path,
                    bot_id,
                    enhancements or [],
                    error_ids,
                )
                try:
                    self.deployer.db.add_trial(bot_id, dep_id, status="active")
                except Exception as exc:
                    self.logger.error("trial record failed: %s", exc)
                if model_id is not None:
                    try:
                        update_model(model_id, current_status="active", db_path=DB_PATH)
                    except Exception as exc:
                        self.logger.error("model update failed: %s", exc)
                    if self.deployer.menace_db:
                        try:
                            self.deployer.menace_db.set_model_status(model_id, "active")
                        except Exception as db_exc:
                            self.logger.error("menace status update failed: %s", db_exc)
                if contrarian_id is not None and self.deployer.contrarian_db:
                    try:
                        self.deployer.contrarian_db.update_status(
                            contrarian_id, "active"
                        )
                    except Exception as exc:
                        self.logger.error("contrarian status update failed: %s", exc)

        if contrarian_id is not None:
            self._update_contrarian_strategy(
                contrarian_id,
                wf_ids,
                list(enhancements or []),
                error_ids,
            )
        self.data_bot.collect(
            bot=self.name,
            response_time=time.time() - start_time,
            errors=len(error_ids),
            tests_failed=0,
            tests_run=0,
            revenue=0.0,
            expense=0.0,
        )
        return ids


__all__ = ["CreationConfig", "BotCreationBot"]
