"""Resource Allocation Bot for dynamic Stage 5 optimisation."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import uuid

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .resource_prediction_bot import ResourceMetrics, TemplateDB
from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from typing import TYPE_CHECKING
from .retry_utils import retry

from .prediction_manager_bot import PredictionManager
from .databases import MenaceDB
from .contrarian_db import ContrarianDB
from db_router import GLOBAL_ROUTER, init_db_router
from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder, FallbackResult, ErrorResult
except Exception as exc:  # pragma: no cover - explicit failure
    raise ImportError(
        "vector_service is required for ResourceAllocationBot; "
        "install via `pip install vector_service`"
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .resources_bot import ResourcesBot
    from .contrarian_model_bot import ContrarianModelBot
    from .bot_database import BotDB


@dataclass
class AllocationRecord:
    """Record of allocation decision."""

    bot: str
    roi: float
    active: bool
    ts: str = datetime.utcnow().isoformat()


router = GLOBAL_ROUTER or init_db_router("resource_allocation")


class AllocationDB:
    """SQLite-backed store for allocation history."""

    def __init__(self, path: Path | str = "allocation.db") -> None:
        # allow connection reuse across threads as the allocator may be invoked
        # from different workers
        self.conn = router.get_connection("allocations")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allocations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot TEXT,
                roi REAL,
                active INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: AllocationRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO allocations(bot, roi, active, ts) VALUES (?, ?, ?, ?)",
            (rec.bot, rec.roi, int(rec.active), rec.ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def history(self) -> pd.DataFrame:
        return pd.read_sql("SELECT bot, roi, active, ts FROM allocations", self.conn)


@self_coding_managed
class ResourceAllocationBot:
    """Manage resources and optimise ROI across bots."""

    prediction_profile = {"scope": ["resources"], "risk": ["medium"]}

    def __init__(
        self,
        db: AllocationDB | None = None,
        template_db: TemplateDB | None = None,
        *,
        context_builder: "ContextBuilder",
        data_bot: "DataBot" | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        prediction_manager: "PredictionManager" | None = None,
        resources_bot: "ResourcesBot" | None = None,
        contrarian_bot: "ContrarianModelBot" | None = None,
        menace_db: "MenaceDB" | None = None,
        contrarian_db: "ContrarianDB" | None = None,
        bot_db: "BotDB" | None = None,
    ) -> None:
        if context_builder is None:
            raise ValueError("ContextBuilder is required")
        self.db = db or AllocationDB()
        self.template_db = template_db or TemplateDB()
        self.context_builder = context_builder
        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:
            logging.getLogger("ResourceAllocationBot").error(
                "context builder refresh failed: %s", exc
            )
            raise RuntimeError("context builder refresh failed") from exc
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.prediction_manager = prediction_manager
        self.resources_bot = resources_bot
        self.contrarian_bot = contrarian_bot
        self.menace_db = menace_db
        self.contrarian_db = contrarian_db
        self.bot_db = bot_db
        self.active_bots: Dict[str, bool] = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ResourceAllocationBot")
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                self.logger.exception("Failed to assign prediction bots: %s", exc)

    # ------------------------------------------------------------------
    @retry(Exception, attempts=3)
    def _record_enhancement(
        self,
        model_id: int,
        new_roi: float,
        old_roi: float,
        *,
        reason: str = "contrarian upgrade",
        triggered_by: str = "ContrarianModelBot",
    ) -> None:
        """Store an enhancement entry for a model replacement."""
        if not self.menace_db:
            return
        try:
            delta = (new_roi - old_roi) / (abs(new_roi) + abs(old_roi) + 1e-6)
            with self.menace_db.engine.begin() as conn:
                menace_id = router.menace_id
                res = conn.execute(
                    "INSERT INTO enhancements("
                    "source_menace_id, description_of_change, reason_for_change, performance_delta, timestamp, triggered_by"
                    ") VALUES (?,?,?,?,?,?)",
                    (
                        menace_id,
                        f"Model {model_id} upgraded via contrarian strategy",
                        reason,
                        delta,
                        datetime.utcnow().isoformat(),
                        triggered_by,
                    ),
                )
                enh_id = int(res.lastrowid if hasattr(res, "lastrowid") else res.inserted_primary_key[0])
                conn.execute(
                    self.menace_db.enhancement_models.insert().values(
                        enhancement_id=enh_id,
                        model_id=model_id,
                    )
                )
                if self.bot_db:
                    cur = self.bot_db.conn.execute(
                        "SELECT bot_id FROM bot_model WHERE model_id=?",
                        (model_id,),
                    )
                    bot_ids = [r[0] for r in cur.fetchall()]
                    for bid in bot_ids:
                        conn.execute(
                            self.menace_db.enhancement_bots.insert().values(
                                enhancement_id=enh_id,
                                bot_id=bid,
                            )
                        )
                        curw = self.bot_db.conn.execute(
                            "SELECT workflow_id FROM bot_workflow WHERE bot_id=?",
                            (bid,),
                        )
                        for wid, in curw.fetchall():
                            conn.execute(
                                self.menace_db.enhancement_workflows.insert().values(
                                    enhancement_id=enh_id,
                                    workflow_id=wid,
                                )
                            )
        except Exception:
            self.logger.exception("Failed to record enhancement for model %s", model_id)
            raise

    # ------------------------------------------------------------------
    def _apply_prediction_bots(self, base: float, metrics: ResourceMetrics) -> float:
        """Combine predictions from assigned bots for ROI adjustment."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        vec = [metrics.cpu, metrics.memory, metrics.disk, metrics.time]
        for pid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(pid)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(vec)
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue
        return float(score / count)

    def evaluate(self, metrics: Dict[str, ResourceMetrics]) -> Dict[str, float]:
        """Estimate ROI using resource consumption and execution time."""
        roi: Dict[str, float] = {}
        for bot, m in metrics.items():
            cost = m.cpu + m.memory / 100 + m.disk
            score = 1.0 / ((cost or 1.0) * max(m.time, 1.0))
            if self.capital_bot:
                score += self.capital_bot.bot_roi(bot) / 1000.0
            if self.data_bot:
                try:
                    score += self.data_bot.roi(bot) / 1000.0
                except Exception:
                    self.logger.warning("ROI lookup failed for %s", bot, exc_info=True)
            score = self._apply_prediction_bots(score, m)
            if self.resources_bot:
                try:
                    ext = self.resources_bot.assess_roi({bot: m})
                    score += ext.get(bot, 0.0)
                except Exception:
                    self.logger.warning("External ROI assessment failed for %s", bot, exc_info=True)
            roi[bot] = score
        return roi

    def allocate(
        self,
        metrics: Dict[str, ResourceMetrics],
        *,
        model_map: Dict[str, int] | None = None,
        weights: Dict[str, float] | None = None,
    ) -> List[Tuple[str, bool]]:
        """Allocate resources based on ROI and deactivate poor performers.

        The optional ``weights`` mapping can boost or penalise the ROI score for
        specific bots. A weight greater than ``1`` will increase the likelihood
        that a bot remains active.
        """
        scores = self.evaluate(metrics)
        if weights:
            for bot, w in weights.items():
                if bot in scores:
                    try:
                        scores[bot] *= float(w)
                    except Exception:
                        continue
        actions: List[Tuple[str, bool]] = []
        for bot, score in scores.items():
            active = score >= 0.1
            self.active_bots[bot] = active
            self.db.add(AllocationRecord(bot=bot, roi=score, active=active))
            if not active:
                self.logger.info("Deactivating %s due to low ROI", bot)
                if model_map and bot in model_map and self.menace_db:
                    try:
                        self.menace_db.set_model_status(model_map[bot], "invalid")
                    except Exception:
                        self.logger.exception("Failed to mark model %s invalid", bot)
                        raise
                    if self.contrarian_db:
                        try:
                            for rec in self.contrarian_db.fetch():
                                if model_map[bot] in self.contrarian_db.models_for(rec.contrarian_id):
                                    self.contrarian_db.update_status(rec.contrarian_id, "invalid")
                        except Exception:
                            self.logger.exception("Failed to update contrarian status for %s", bot)
                            raise
                if self.bot_db:
                    record = self.bot_db.find_by_name(bot)
                    if record:
                        bid = record["id"]
                        try:
                            cur = self.bot_db.conn.execute(
                                "SELECT COUNT(*) FROM bot_model WHERE bot_id=?",
                                (bid,),
                            )
                            cnt = cur.fetchone()[0]
                        except Exception:
                            cnt = 0
                        if cnt <= 1:
                            try:
                                self.bot_db.update_bot(bid, status="inactive")
                            except Exception:
                                self.logger.exception("Failed to deactivate bot %s", bot)
                                raise
            if self.contrarian_bot and score < 0.05:
                try:
                    innov = self.contrarian_bot.activate_on_roi_drop(score)
                    if innov and model_map and bot in model_map:
                        if innov.roi > score:
                            self._record_enhancement(model_map[bot], innov.roi, score)
                except Exception:
                    self.logger.warning("Contrarian bot activation failed for %s", bot, exc_info=True)
            actions.append((bot, active))
        return actions

    def suggest_improvement(self, bot: str) -> str:
        builder = self.context_builder
        if not isinstance(builder, ContextBuilder):
            self.logger.error("context_builder is required for improvement prompts")
            return "upgrade"
        session_id = uuid.uuid4().hex
        try:
            ctx_res = builder.build(bot, session_id=session_id)
            context = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
            if isinstance(context, (FallbackResult, ErrorResult)):
                context = ""
            elif context:
                context = compress_snippets({"snippet": context}).get("snippet", context)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("Context build failed for %s", bot)
            return "upgrade"
        try:
            from prompt_engine import PromptEngine
            from local_backend import mixtral_client
        except Exception:  # pragma: no cover - dependencies missing
            return "upgrade"
        llm = None
        try:  # pragma: no cover - optional local backend
            llm = mixtral_client()
        except Exception:
            pass
        engine = PromptEngine(context_builder=builder, llm=llm)
        prompt = engine.build_prompt(
            f"Improve {bot}",
            retrieval_context=context,
            context_builder=builder,
        )
        prompt.metadata["retrieval_session_id"] = session_id
        if engine.llm is None:
            return "upgrade"
        try:
            result = engine.llm.generate(prompt, context_builder=builder)
            return (getattr(result, "text", "") or "").strip()
        except Exception:  # pragma: no cover - local LLM failures
            self.logger.exception("Improvement suggestion failed for %s", bot)
            return "upgrade"

    def genetic_step(self, strategies: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Select strategy with highest ROI."""
        best = max(strategies, key=lambda s: s.get("roi", 0.0))
        self.logger.info("Selected strategy %s", best)
        return best

    def log_to_prediction(self, bot: str, metrics: ResourceMetrics) -> None:
        self.template_db.add(bot, metrics)
        self.template_db.save()


__all__ = ["AllocationRecord", "AllocationDB", "ResourceAllocationBot"]
