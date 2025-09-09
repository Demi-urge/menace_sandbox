from __future__ import annotations

"""Central orchestrator coordinating all Menace stages and oversight bots."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable
import logging
import os
import time
import threading
import asyncio
import uuid
from dynamic_path_router import resolve_path, get_project_root
from .knowledge_graph import KnowledgeGraph

from .advanced_error_management import AutomatedRollbackManager
from .self_coding_engine import SelfCodingEngine
from .rollback_validator import RollbackValidator

from .oversight_bots import (
    L1OversightBot,
    L2OversightBot,
    L3OversightBot,
    M1OversightBot,
    M2OversightBot,
    M3OversightBot,
    H1OversightBot,
    H2OversightBot,
    H3OversightBot,
)
from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
from .discrepancy_detection_bot import DiscrepancyDetectionBot
from .efficiency_bot import EfficiencyBot
from .neuroplasticity import Outcome, PathwayDB, PathwayRecord
from .ad_integration import AdIntegration
from .watchdog import Watchdog, ContextBuilder
from .error_bot import ErrorDB
from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .trending_scraper import TrendingScraper
from .self_learning_service import main as learning_main
from .strategic_planner import StrategicPlanner
from .strategy_prediction_bot import StrategyPredictionBot
from .autoscaler import Autoscaler
from .trend_predictor import TrendPredictor
from .identity_seeder import seed_identity
from .session_vault import SessionVault
import requests
from .cognition_layer import build_cognitive_context, log_feedback
import db_router
from db_router import DBRouter


class _RemoteVisualAgent:
    """Minimal client polling the remote visual agent service."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        poll_interval: float | None = None,
    ) -> None:
        default_url = os.getenv("VISUAL_DESKTOP_URL", "http://127.0.0.1:8001")
        self.url = (url or default_url).rstrip("/")
        self.token = token or os.getenv("VISUAL_AGENT_TOKEN", "")
        self.poll_interval = poll_interval or float(
            os.getenv("VISUAL_AGENT_POLL_INTERVAL", "5")
        )
        self.logger = logging.getLogger("RemoteVisualAgent")

    def _poll_status(self, tid: str) -> str:
        while True:
            try:
                resp = requests.get(f"{self.url}/status/{tid}", timeout=10)
                if resp.status_code == 200:
                    status = resp.json().get("status", "")
                    if status not in {"queued", "running"}:
                        return status
            except Exception as exc:
                self.logger.warning("status poll failed for %s: %s", tid, exc)
            time.sleep(self.poll_interval)

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, object]:
        prompt = "\n".join(m.get("content", "") for m in messages)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            resp = requests.post(
                f"{self.url}/run",
                json={"prompt": prompt, "branch": None},
                headers=headers,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"visual agent request failed: {exc}") from exc

        if resp.status_code == 202:
            data = resp.json()
            tid = data.get("id")
            status = data.get("status", "")
            if tid:
                status = self._poll_status(str(tid))
            return {"choices": [{"message": {"content": status}}]}
        if resp.status_code == 409:
            # wait until idle then retry
            while True:
                try:
                    sresp = requests.get(f"{self.url}/status", timeout=10)
                    if sresp.status_code == 200 and not sresp.json().get("active", False):
                        break
                except Exception:
                    pass
                time.sleep(self.poll_interval)
            return self.ask(messages)
        raise RuntimeError(f"status {resp.status_code}: {resp.text}")

    def revert(self) -> bool:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            resp = requests.post(f"{self.url}/revert", headers=headers, timeout=10)
            return resp.status_code == 202
        except Exception as exc:
            self.logger.warning("revert failed: %s", exc)
            return False


class _SimpleScheduler:
    """Very small in-process scheduler used when APScheduler is unavailable."""

    def __init__(self) -> None:
        self.tasks: dict[str, tuple[float, Callable[[], None], float]] = {}
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger("SimpleScheduler")

    def add_job(self, func: Callable[[], None], interval: float, id: str) -> None:
        """Schedule *func* every *interval* seconds."""
        with self.lock:
            self.tasks[id] = (interval, func, time.time() + interval)
        if not self.thread:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def remove_job(self, id: str) -> None:
        """Remove a job by *id* if present."""
        with self.lock:
            self.tasks.pop(id, None)

    def reschedule_job(self, id: str, interval: float) -> None:
        """Change job *interval* and reset its next run time."""
        with self.lock:
            job = self.tasks.get(id)
            if job:
                _, func, _ = job
                self.tasks[id] = (interval, func, time.time() + interval)

    def _run(self) -> None:
        while not self.stop.is_set():
            now = time.time()
            with self.lock:
                jobs = list(self.tasks.items())
            for jid, (interval, fn, nxt) in jobs:
                if now >= nxt:
                    try:
                        fn()
                    except Exception:
                        self.logger.exception("job %s failed", jid)
                    with self.lock:
                        if jid in self.tasks:
                            self.tasks[jid] = (interval, fn, time.time() + interval)
            with self.lock:
                next_times = [n for _, (_, _, n) in self.tasks.items()]
            sleep_for = 0.1
            if next_times:
                sleep_for = max(0.0, min(next_times) - time.time())
            self.stop.wait(min(sleep_for, 1.0))

    def shutdown(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=0)


try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - APScheduler missing
    BackgroundScheduler = None  # type: ignore


@dataclass
class BotNode:
    """Simple representation of a bot within the hierarchy."""

    name: str
    level: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class MenaceOrchestrator:
    """Drive the entire Menace workflow using a hierarchical structure."""

    def __init__(
        self,
        pathway_db: PathwayDB | None = None,
        myelination_threshold: float = 1.0,
        ad_client: AdIntegration | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        menace_id: str | None = None,
        router: DBRouter | None = None,
        *,
        context_builder: ContextBuilder,
        on_restart: Callable[[str], None] | None = None,
        auto_bootstrap: bool | None = None,
        visual_agent_client: object | None = None,
    ) -> None:
        menace_id = menace_id or uuid.uuid4().hex
        self.menace_id = menace_id
        if router is None:
            router = DBRouter(
                menace_id,
                str(resolve_path(f"menace_{menace_id}_local.db")),
                str(resolve_path("shared/global.db")),
            )
        db_router.GLOBAL_ROUTER = router
        self.router = router
        self.context_builder = context_builder
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MenaceOrchestrator")
        try:
            self.context_builder.refresh_db_weights()
        except Exception:
            self.logger.exception("context builder refresh failed")
        self.pipeline = ModelAutomationPipeline(
            pathway_db=pathway_db,
            myelination_threshold=myelination_threshold,
            context_builder=self.context_builder,
        )
        self.pathway_db = pathway_db
        self.myelination_threshold = myelination_threshold
        self.ad_client = ad_client or AdIntegration()
        self.rollback_mgr = rollback_mgr
        if auto_bootstrap is None:
            auto_bootstrap = not bool(os.getenv("MENACE_LIGHT_IMPORTS"))
        self.model_id: int | None = None
        if auto_bootstrap:
            try:
                from .self_model_bootstrap import bootstrap as self_bootstrap

                self.model_id = self_bootstrap(
                    context_builder=self.context_builder
                )
            except Exception:
                self.logger.exception("self bootstrap failed")
        self.restart_callback = on_restart
        self.engines: Dict[str, SelfCodingEngine] = {}
        self._patches: Dict[str, List[str]] = {}
        self.nodes: Dict[str, BotNode] = {}
        self.oversight: Dict[str, Dict[str, object]] = {
            "L1": {},
            "L2": {},
            "L3": {},
            "M1": {},
            "M2": {},
            "M3": {},
            "H1": {},
            "H2": {},
            "H3": {},
        }
        self._classes = {
            "L1": L1OversightBot,
            "L2": L2OversightBot,
            "L3": L3OversightBot,
            "M1": M1OversightBot,
            "M2": M2OversightBot,
            "M3": M3OversightBot,
            "H1": H1OversightBot,
            "H2": H2OversightBot,
            "H3": H3OversightBot,
        }
        self.knowledge_graph = KnowledgeGraph()
        self.bot_status: Dict[str, bool] = {}
        self.last_root_causes: Dict[str, List[str]] = {}
        self.workflow_confidence: Dict[str, float] = {}
        self._workflow_counts: Dict[str, int] = {}
        self.watchdog = Watchdog(
            ErrorDB(router=self.router),
            ROIDB(),
            MetricsDB(),
            context_builder=self.context_builder,
        )
        self.scheduler: object | None = None
        self.planner = StrategicPlanner(
            StrategyPredictionBot(), Autoscaler(), TrendPredictor()
        )
        self.last_plan: str | None = None
        self.discrepancy_detector = DiscrepancyDetectionBot()
        self.bottleneck_detector = EfficiencyBot()
        self.visual_client = visual_agent_client or _RemoteVisualAgent()

    # ------------------------------------------------------------------
    def status_summary(self) -> Dict[str, object]:
        """Return a high level status overview for dashboards."""
        active = [n for n, ok in self.bot_status.items() if ok]
        failed = [n for n, ok in self.bot_status.items() if not ok]
        return {
            "active": active,
            "failed": failed,
            "last_plan": self.last_plan,
            "workflow_confidence": dict(self.workflow_confidence),
        }

    # ------------------------------------------------------------------
    def set_restart_callback(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to run after any job restart."""
        self.restart_callback = cb

    # ------------------------------------------------------------------
    def create_oversight(
        self,
        name: str,
        level: str,
        parent: Optional[str] = None,
        dependencies: Optional[Iterable[str]] = None,
    ) -> None:
        """Instantiate an oversight bot and register it."""
        if name in self.nodes:
            return
        cls = self._classes.get(level)
        if not cls:
            raise ValueError(f"invalid level {level}")
        bot = cls()
        self.oversight[level][name] = bot
        self.nodes[name] = BotNode(
            name=name,
            level=level,
            parent=parent,
            dependencies=list(dependencies or []),
        )
        self.bot_status[name] = True
        if parent and parent in self.nodes:
            self.nodes[parent].children.append(name)

    def hierarchy(self, level: str) -> Dict[str, object]:
        """Return oversight bots for a level."""
        return self.oversight.get(level, {})

    def register_engine(self, name: str, engine: SelfCodingEngine) -> None:
        """Associate a :class:`SelfCodingEngine` with an oversight node."""
        if getattr(engine, "llm_client", None) is None and self.visual_client:
            engine.llm_client = self.visual_client
        self.engines[name] = engine

    # ------------------------------------------------------------------
    def check_health(self, name: str) -> List[str]:
        """Query the knowledge graph for root causes of a failed bot."""
        causes = []
        try:
            causes = self.knowledge_graph.root_causes(name)
        except Exception:
            causes = []
        self.last_root_causes[name] = causes
        return causes

    def record_failure(self, name: str) -> List[str]:
        """Mark ``name`` as failed and run health checks."""
        self.bot_status[name] = False
        return self.check_health(name)

    def handle_feedback(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
        errors: Iterable[Dict[str, Any]] | None = None,
    ) -> None:
        """Record patch outcome and forward failures to :class:`PatchSafety`.

        ``session_id`` identifies the retrieval session, ``success`` indicates
        whether the patch ultimately succeeded and ``contribution`` captures the
        outcome score or ROI contribution.  When ``errors`` are supplied and the
        patch failed, each entry is forwarded to
        :func:`PatchSafety.record_failure` so the ``failure_vectorizer`` can
        learn from the example.
        """

        log_feedback(
            session_id,
            success,
            patch_id=patch_id,
            contribution=contribution,
            context_builder=self.context_builder,
        )
        if not success and errors:
            patch_safety = getattr(self.context_builder, "patch_safety", None)
            for err in errors:
                if patch_safety is not None:
                    try:
                        patch_safety.record_failure(dict(err))
                    except Exception:
                        self.logger.exception("failed to record failure metadata")

    def update_confidence_metrics(self, results: Dict[str, bool]) -> None:
        """Update workflow confidence scores based on replay results."""
        for wf, ok in results.items():
            prev = self.workflow_confidence.get(wf, 0.0)
            count = self._workflow_counts.get(wf, 0)
            new = (prev * count + (1.0 if ok else 0.0)) / (count + 1)
            self.workflow_confidence[wf] = new
            self._workflow_counts[wf] = count + 1

    def reassign_task(
        self,
        bot: str,
        dependencies: Iterable[str] | None = None,
        alternatives: Iterable[str] | None = None,
    ) -> str:
        """Return ``bot`` or a fallback if dependencies are unavailable."""
        if any(not self.bot_status.get(d, True) for d in (dependencies or [])):
            for alt in alternatives or []:
                if self.bot_status.get(alt, True):
                    return alt
        return bot

    def apply_patch_all(
        self, node_paths: Dict[str, Path], description: str
    ) -> Dict[str, tuple[int | None, bool]]:
        """Apply the same patch to multiple nodes and rollback if any fails."""
        results: Dict[str, tuple[int | None, bool]] = {}
        rollback_validator: RollbackValidator | None = None
        if self.rollback_mgr:
            try:
                rollback_validator = RollbackValidator(self.rollback_mgr.path)
            except Exception:
                rollback_validator = None
        failure = False
        failing_pid: str | None = None
        any_success = False
        for node, path in node_paths.items():
            eng = self.engines.get(node)
            if not eng:
                continue
            _ctx, session_id = build_cognitive_context(
                f"apply patch {description} on {node}",
                context_builder=self.context_builder,
            )
            try:
                pid, reverted, _ = eng.apply_patch(
                    path,
                    description,
                    reason=description,
                    trigger="menace_orchestrator",
                )
                results[node] = (pid, reverted)
                success = pid is not None and not reverted
                self.handle_feedback(
                    session_id,
                    success,
                    patch_id=str(pid) if pid is not None else "",
                    contribution=1.0 if success else 0.0,
                )
                any_success = any_success or success
            except Exception:
                self.handle_feedback(session_id, False, contribution=0.0)
                raise
            if pid is not None and not reverted and self.rollback_mgr:
                self.rollback_mgr.register_patch(str(pid), node)
            if reverted or pid is None:
                self.record_failure(node)
                failure = True
                if failing_pid is None and pid is not None:
                    failing_pid = str(pid)
        if failure:
            nodes = list(node_paths.keys())
            if failing_pid and self.rollback_mgr:
                self.rollback_mgr.auto_rollback(failing_pid, nodes)
                if rollback_validator and not rollback_validator.verify_rollback(
                    failing_pid, nodes
                ):
                    self.logger.error(
                        "rollback verification failed for %s on nodes %s",
                        failing_pid,
                        ",".join(nodes),
                    )
            for node, (pid, _) in results.items():
                eng = self.engines.get(node)
                if not eng:
                    continue
                if pid is not None and self.rollback_mgr:
                    self.rollback_mgr.rollback(str(pid))
                if pid is not None:
                    eng.rollback_patch(str(pid))
        if any_success:
            try:
                from sandbox_runner import discover_recursive_orphans
                import sandbox_runner.environment as environment

                repo = Path(get_project_root())
                mapping = discover_recursive_orphans(str(repo))
                modules = set(mapping.keys())
                if modules:
                    added_modules: set[str] = set()
                    try:
                        _, tested = environment.auto_include_modules(
                            sorted(modules),
                            recursive=True,
                            context_builder=self.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                    except Exception:
                        self.logger.exception("auto_include_modules failed")
                        added_modules = set()
                    if added_modules:
                        try:
                            grapher = getattr(self, "module_synergy_grapher", None)
                            if grapher is None:
                                from module_synergy_grapher import ModuleSynergyGrapher

                                grapher = ModuleSynergyGrapher(root=repo)
                                graph_path = resolve_path("sandbox_data/module_synergy_graph.json")
                                try:
                                    grapher.load(graph_path)
                                except Exception:
                                    pass
                                self.module_synergy_grapher = grapher
                            grapher.update_graph(sorted(added_modules))
                        except Exception:
                            self.logger.exception("failed to update synergy graph")
                        try:
                            clusterer = getattr(self, "intent_clusterer", None)
                            if clusterer is None:
                                from intent_clusterer import IntentClusterer

                                data_dir = resolve_path("sandbox_data")
                                clusterer = IntentClusterer(
                                    local_db_path=data_dir / "intent.db",
                                    shared_db_path=data_dir / "intent.db",
                                )
                                self.intent_clusterer = clusterer
                            paths = {
                                Path(resolve_path(repo / f"{m}.py"))
                                for m in added_modules
                            }
                            clusterer.index_modules(paths)
                        except Exception:
                            self.logger.exception("failed to index intent modules")
            except Exception:
                self.logger.exception("recursive orphan auto inclusion failed")
        return results

    # ------------------------------------------------------------------
    def run_cycle(self, models: Iterable[str]) -> Dict[str, AutomationResult]:
        """Execute a full pipeline run for each model."""
        return asyncio.run(self.run_cycle_async(models))

    async def run_cycle_async(self, models: Iterable[str]) -> Dict[str, AutomationResult]:
        """Async wrapper running the pipeline and ad processing concurrently."""
        results: Dict[str, AutomationResult] = {}
        scored: List[tuple[str, float]] = []
        for model in models:
            score = 0.0
            if self.pathway_db:
                sim = self.pathway_db.similar_actions(f"run_cycle:{model}", limit=1)
                if sim:
                    score = sim[0][1]
            scored.append((model, score))
        if self.pathway_db:
            scored.sort(key=lambda x: x[1], reverse=True)
            models = [m for m, _ in scored]
        run_ids: List[int] = []
        queue: List[str] = list(models)
        processed: List[str] = []
        while queue:
            model = queue.pop(0)
            pipeline_task = asyncio.to_thread(self.pipeline.run, model)
            ad_task = self.ad_client.process_sales_async()
            res, _ = await asyncio.gather(pipeline_task, ad_task, return_exceptions=True)
            if isinstance(res, Exception):
                raise res
            results[model] = res
            if self.pathway_db:
                outcome = Outcome.SUCCESS if res.package else Outcome.FAILURE
                roi_val = res.roi.roi if res.roi else 0.0
                pid = self.pathway_db.log(
                    PathwayRecord(
                        actions=f"run_cycle:{model}",
                        inputs=model,
                        outputs=str(res.package),
                        exec_time=0.0,
                        resources="",
                        outcome=outcome,
                        roi=roi_val,
                    )
                )
                run_ids.append(pid)

                next_pid = self.pathway_db.next_pathway(pid)
                if next_pid and self.pathway_db.is_highly_myelinated(
                    next_pid, self.myelination_threshold
                ):
                    row = self.pathway_db.conn.execute(
                        "SELECT actions FROM pathways WHERE id=?",
                        (next_pid,),
                    ).fetchone()
                    if row:
                        actions = row[0]
                        if actions.startswith("run_cycle:"):
                            next_model = actions.split("run_cycle:", 1)[1].split("->")[0]
                            if next_model not in processed and next_model not in queue:
                                queue.insert(0, next_model)
            processed.append(model)
        if self.pathway_db and run_ids:
            self.pathway_db.record_sequence(run_ids)
            self.pathway_db.merge_macro_pathways()
        # run post-cycle diagnostics
        try:
            detections = self.discrepancy_detector.scan()
            self.logger.info("discrepancy findings: %s", detections)
            if any(
                d.severity >= self.discrepancy_detector.severity_threshold * 2
                for d in detections
            ):
                self.pipeline.roi_threshold += 0.1
        except Exception:
            self.logger.exception("discrepancy detector failed")
        try:
            report = self.bottleneck_detector.assess_efficiency()
            self.logger.info("bottleneck report: %s", report)
            if report.get("predicted_bottleneck", 0.0) > 0.8:
                self.pipeline.roi_threshold += 0.1
        except Exception:
            self.logger.exception("bottleneck detection failed")
        return results

    # ------------------------------------------------------------------
    def receive_scaling_hint(self, hint: str) -> None:
        """React to scaling recommendations from external allocators."""
        self.logger.info("scaling hint: %s", hint)
        if hint in {"scale_up", "scale_down"}:
            try:
                metrics = {"cpu": 0.9, "memory": 0.9}
                if hint == "scale_down":
                    metrics = {"cpu": 0.1, "memory": 0.1}
                self.planner.autoscaler.scale(metrics)
            except Exception:
                self.logger.exception("autoscaler failed")
        else:
            try:
                bots = list(self.engines.keys()) or list(self.nodes.keys())
                self.pipeline.allocator.allocate(bots)
            except Exception:
                self.logger.exception("allocator failed")

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------
    def _heartbeat(self, name: str) -> None:
        try:
            self.watchdog.record_heartbeat(name)
        except Exception:
            self.logger.exception("heartbeat failed for %s", name)

    def _trending_job(self) -> None:
        try:
            TrendingScraper().scrape_reddit()
        except Exception:
            self.logger.exception("trending job failed")
        self._heartbeat("trending_scan")

    def _learning_job(self) -> None:
        try:
            learning_main(stop_event=threading.Event())
        except Exception:
            self.logger.exception("learning job failed")
        self._heartbeat("learning")

    def _seed_job(self) -> None:
        url = os.getenv("SIGNUP_URL")
        if not url:
            self.logger.warning("SIGNUP_URL not set, skipping seed job")
            return
        vault_path = os.getenv("VAULT_PATH", "sessions.db")
        threshold = int(os.getenv("IDENTITY_STOCK_THRESHOLD", "5"))
        vault = SessionVault(path=vault_path)
        domain = url.split("//")[-1].split("/")[0]
        try:
            count = vault.count(domain)
        except Exception:
            self.logger.exception("vault count failed")
            count = threshold
        if count < threshold:
            try:
                seed_identity(url, vault)
            except Exception:
                self.logger.exception("seed job failed")
        self._heartbeat("identity_seed")

    def _planning_job(self) -> None:
        try:
            self.last_plan = self.planner.plan_cycle()
        except Exception:
            self.logger.exception("planning job failed")
        self._heartbeat("planning")

    def restart_job(self, job_id: str) -> None:
        if self.scheduler is None:
            return
        try:
            if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
                try:
                    self.scheduler.remove_job(job_id)
                except Exception:
                    self.logger.exception("failed to remove job %s", job_id)
            elif isinstance(self.scheduler, _SimpleScheduler):
                self.scheduler.remove_job(job_id)
            else:
                return
        except Exception:
            self.logger.exception("scheduler restart failed")

        interval_map = {
            "trending_scan": int(os.getenv("TRENDING_SCAN_INTERVAL", "3600")),
            "learning": int(os.getenv("LEARNING_INTERVAL", "7200")),
            "planning": int(os.getenv("PLANNING_INTERVAL", "3600")),
            "identity_seed": int(os.getenv("IDENTITY_SEED_INTERVAL", "3600")),
        }
        func_map = {
            "trending_scan": self._trending_job,
            "learning": self._learning_job,
            "planning": self._planning_job,
            "identity_seed": self._seed_job,
        }
        interval = interval_map.get(job_id)
        func = func_map.get(job_id)
        if interval is None or func is None:
            return
        if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
            self.scheduler.add_job(func, "interval", seconds=interval, id=job_id)
        else:
            self.scheduler.add_job(func, interval, job_id)
        if self.restart_callback:
            try:
                self.restart_callback(job_id)
            except Exception:
                self.logger.exception("restart callback failed")

    def start_scheduled_jobs(self) -> None:
        t_int = int(os.getenv("TRENDING_SCAN_INTERVAL", "3600"))
        l_int = int(os.getenv("LEARNING_INTERVAL", "7200"))
        p_int = int(os.getenv("PLANNING_INTERVAL", "3600"))
        s_int = int(os.getenv("IDENTITY_SEED_INTERVAL", "3600"))
        if BackgroundScheduler:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(
                self._trending_job, "interval", seconds=t_int, id="trending_scan"
            )
            self.scheduler.add_job(
                self._learning_job, "interval", seconds=l_int, id="learning"
            )
            self.scheduler.add_job(
                self._planning_job, "interval", seconds=p_int, id="planning"
            )
            self.scheduler.add_job(
                self._seed_job, "interval", seconds=s_int, id="identity_seed"
            )
            self.scheduler.start()
        else:
            self.scheduler = _SimpleScheduler()
            self.scheduler.add_job(self._trending_job, t_int, "trending_scan")
            self.scheduler.add_job(self._learning_job, l_int, "learning")
            self.scheduler.add_job(self._planning_job, p_int, "planning")
            self.scheduler.add_job(self._seed_job, s_int, "identity_seed")
        wd_int = int(os.getenv("WATCHDOG_INTERVAL", "60"))
        self.watchdog.healer.heal = lambda bot, pid=None: self.restart_job(bot)
        self.watchdog.schedule(interval=wd_int)

    def stop_scheduled_jobs(self) -> None:
        if self.scheduler is None:
            return
        if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
            self.scheduler.shutdown(wait=False)
        else:
            self.scheduler.shutdown()

    def shutdown(self) -> None:
        """Clean up resources before exiting."""
        self.stop_scheduled_jobs()


__all__ = ["BotNode", "MenaceOrchestrator"]
