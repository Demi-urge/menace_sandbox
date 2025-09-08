# flake8: noqa
from __future__ import annotations

import json
import time
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Dict, Any, Optional
from dynamic_path_router import resolve_path

from .research_aggregator_bot import ResearchAggregatorBot, InfoDB, ResearchItem
from .chatgpt_enhancement_bot import EnhancementDB
from .prediction_manager_bot import PredictionManager
from .data_bot import DataBot
from .strategy_prediction_bot import StrategyPredictionBot, CompetitorFeatures
from .resource_allocation_bot import ResourceAllocationBot
from vector_service.context_builder import ContextBuilder
from .resource_prediction_bot import ResourceMetrics
from .task_handoff_bot import WorkflowDB as HandoffWorkflowDB, WorkflowRecord
from .contrarian_db import ContrarianDB, ContrarianRecord
from .capital_management_bot import CapitalManagementBot
from .unified_event_bus import UnifiedEventBus

WORKFLOW_DB = resolve_path("contrarian_model_bot/workflows_db.json")
INNOVATIONS_DB = resolve_path("contrarian_model_bot/innovations_db.json")

@dataclass
class WorkflowStep:
    name: str
    description: str = ""
    risk: float = 0.0

@dataclass
class Workflow:
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Workflow:
        """Reconstruct a :class:`Workflow` from :meth:`to_dict` output."""

        steps = [WorkflowStep(**s) for s in data.get("steps", [])]
        return cls(name=data.get("name", ""), steps=steps, tags=data.get("tags", []))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of the workflow."""

        return {
            "name": self.name,
            "steps": [s.__dict__ for s in self.steps],
            "tags": self.tags,
        }

class WorkflowDB:
    def __init__(self, path: Path = WORKFLOW_DB) -> None:
        self.path = path

    def load(self) -> List[Workflow]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except Exception:
                data = []
        else:
            data = []
        return [Workflow.from_dict(w) for w in data]

    def save(self, workflows: Iterable[Workflow]) -> None:
        data = [w.to_dict() for w in workflows]
        self.path.write_text(json.dumps(data, indent=2))

@dataclass
class Innovation:
    name: str
    workflow: Workflow
    risk: float
    roi: float
    timestamp: str = datetime.utcnow().isoformat()

class InnovationsDB:
    def __init__(self, path: Path = INNOVATIONS_DB) -> None:
        self.path = path
        if not self.path.exists():
            self.path.write_text("[]")

    def add(self, innovation: Innovation) -> None:
        items = self.fetch()
        items.append(innovation)
        data = [
            {
                "name": i.name,
                "workflow": i.workflow.to_dict(),
                "risk": i.risk,
                "roi": i.roi,
                "timestamp": i.timestamp,
            }
            for i in items
        ]
        self.path.write_text(json.dumps(data, indent=2))

    def fetch(self) -> List[Innovation]:
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return []
        items = []
        for d in data:
            wf = Workflow.from_dict(d.get("workflow", {}))
            items.append(
                Innovation(
                    name=d.get("name", ""),
                    workflow=wf,
                    risk=float(d.get("risk", 0.0)),
                    roi=float(d.get("roi", 0.0)),
                    timestamp=d.get("timestamp", ""),
                )
            )
        return items

class ContrarianModelBot:
    prediction_profile = {"scope": ["contrarian"], "risk": ["high"]}
    def __init__(
        self,
        workflow_db: Optional[WorkflowDB] = None,
        workflows_db: Optional[HandoffWorkflowDB] = None,
        innovations_db: Optional[InnovationsDB] = None,
        info_db: Optional[InfoDB] = None,
        enhancements_db: Optional[EnhancementDB] = None,
        context_builder: Optional[ContextBuilder] = None,
        aggregator: Optional[ResearchAggregatorBot] = None,
        prediction_manager: Optional[PredictionManager] = None,
        data_bot: Optional[DataBot] = None,
        strategy_bot: Optional[StrategyPredictionBot] = None,
        allocator: Optional[ResourceAllocationBot] = None,
        contrarian_db: Optional[ContrarianDB] = None,
        capital_manager: Optional[CapitalManagementBot] = None,
        risk_tolerance: float = 0.7,
        resources: float = 1.0,
        model_ids: Optional[List[int]] = None,
        event_bus: UnifiedEventBus | None = None,
    ) -> None:
        self.workflow_db = workflow_db or WorkflowDB()
        self.workflows_db = workflows_db or HandoffWorkflowDB(event_bus=event_bus)
        self.innovations_db = innovations_db or InnovationsDB()
        self.info_db = info_db or InfoDB()
        self.enh_db = enhancements_db or EnhancementDB()
        self.context_builder = context_builder
        if self.context_builder is not None:
            self.context_builder.refresh_db_weights()
        if aggregator is None:
            if self.context_builder is None:
                raise ValueError("context_builder is required when aggregator is not provided")
            self.aggregator = ResearchAggregatorBot(
                [],
                info_db=self.info_db,
                enhancements_db=self.enh_db,
                context_builder=self.context_builder,
            )
        else:
            self.aggregator = aggregator
        self.prediction_manager = prediction_manager
        self.data_bot = data_bot
        self.strategy_bot = strategy_bot or StrategyPredictionBot()
        if allocator is None:
            if self.context_builder is None:
                raise ValueError("context_builder is required when allocator is not provided")
            allocator = ResourceAllocationBot(context_builder=self.context_builder)
        self.allocator = allocator
        self.contrarian_db = contrarian_db or ContrarianDB()
        self.capital_manager = capital_manager
        self.model_ids = model_ids or []
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.risk_tolerance = risk_tolerance
        self.resources = resources
        self.energy = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workflows = self.workflow_db.load()

    def _apply_prediction_bots(self, risk: float, reward: float) -> tuple[float, float]:
        if not self.prediction_manager:
            return risk, reward
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if callable(pred):
                try:
                    out = pred([risk, reward])
                    if isinstance(out, (list, tuple)) and len(out) == 2:
                        risk, reward = float(out[0]), float(out[1])
                    elif isinstance(out, (int, float)):
                        reward = float(out)
                except Exception:
                    self.logger.exception("prediction bot %s failed", bot_id)
                    continue
        return risk, reward

    def _score_workflow(self, wf: Workflow) -> float:
        """Return an ROI estimate for a workflow based on multiple factors."""
        risk = sum(step.risk for step in wf.steps) / max(len(wf.steps), 1)
        unique = len(set(t for step in wf.steps for t in step.name.split()))
        avg_desc = sum(len(step.description) for step in wf.steps) / max(len(wf.steps), 1)
        tag_factor = len(wf.tags) * 0.1
        base = (unique * 0.05 + avg_desc * 0.01 + tag_factor)
        return max(base * (1.0 - risk), 0.0)

    def _update_risk_tolerance(self) -> None:
        if not self.capital_manager:
            return
        try:
            energy = self.capital_manager.energy_score(load=0.0, success_rate=1.0, deploy_eff=1.0, failure_rate=0.0)
            self.energy = energy
            self.risk_tolerance = 0.9 - 0.6 * energy
        except Exception:
            self.logger.exception("failed to update risk tolerance")

    def should_experiment(self, risk: float, required: float = 0.2) -> bool:
        return risk >= self.risk_tolerance and self.resources >= required

    def merge_workflows(self, a: Workflow, b: Workflow, name: Optional[str] = None) -> Workflow:
        steps = a.steps + [s for s in b.steps if s.name not in {st.name for st in a.steps}]
        return Workflow(name=name or f"{a.name}+{b.name}", steps=steps, tags=sorted(set(a.tags + b.tags)))

    def allocate_resources(self, roi: float) -> Dict[str, ResourceMetrics]:
        metrics = {"contrarian": ResourceMetrics(cpu=roi, memory=roi * 10, disk=1.0, time=1.0)}
        try:
            self.allocator.allocate(metrics)
        except Exception:
            self.logger.exception("resource allocation failed")
        return metrics

    def ideate(self) -> Optional[Innovation]:
        if len(self.workflows) < 2:
            return None
        base, other = self.workflows[0], self.workflows[1]
        self._update_risk_tolerance()
        hybrid = self.merge_workflows(base, other)
        risk = sum(s.risk for s in hybrid.steps) / max(len(hybrid.steps), 1)
        roi = self._score_workflow(hybrid)
        risk, roi = self._apply_prediction_bots(risk, roi)
        if not self.should_experiment(risk):
            return None
        innov = Innovation(name=hybrid.name, workflow=hybrid, risk=risk, roi=roi)
        self.innovations_db.add(innov)
        metrics = self.allocate_resources(roi)
        cid = None
        if self.contrarian_db:
            try:
                rec = ContrarianRecord(
                    innovation_name=innov.name,
                    innovation_type="workflow",
                    risk_score=risk,
                    reward_score=roi,
                    activation_trigger="ideate",
                    resource_allocation=metrics["contrarian"].__dict__,
                )
                cid = self.contrarian_db.add(rec)
                for mid in self.model_ids:
                    self.contrarian_db.link_model(cid, mid)
            except Exception:
                self.logger.exception("contrarian db update failed")
        try:
            for mid in self.model_ids:
                self.aggregator.info_db.set_current_model(mid)
                if cid is not None:
                    self.aggregator.info_db.set_current_contrarian(cid)
            self.aggregator.requirements = [innov.name]
            self.aggregator.process(innov.name, energy=max(1, int(round(1 + self.energy * 2))))
        except Exception:
            self.logger.exception("research aggregator failed")
        try:
            self.workflows_db.add(WorkflowRecord(
                workflow=[s.name for s in hybrid.steps],
                title=hybrid.name,
                description="generated by contrarian bot",
                task_sequence=[s.name for s in hybrid.steps],
                tags=hybrid.tags,
                category="contrarian",
                type_="workflow",
            ))
        except Exception:
            self.logger.exception("workflow logging failed")
        try:
            record = ResearchItem(
                topic=innov.name,
                content=json.dumps(hybrid.to_dict()),
                timestamp=time.time(),
                title=innov.name,
                tags=hybrid.tags,
                category="workflow",
                type_="contrarian",
                associated_bots=["ContrarianModelBot"],
                model_id=self.model_ids[0] if self.model_ids else None,
                contrarian_id=cid or 0,
            )
            self.info_db.add(record)
        except Exception:
            self.logger.exception("info db update failed")
        return innov

    def activate_on_roi_drop(self, roi: float, threshold: float = 0.0, model_name: Optional[str] = None) -> Optional[Innovation]:
        if self.data_bot and model_name:
            try:
                df = self.data_bot.db.fetch(limit=50)
                df = df[df["bot"] == model_name]
                if not df.empty and "roi" in df.columns:
                    roi = float(df["roi"].iloc[0])
            except Exception:
                self.logger.exception("ROI lookup failed")
        return self.ideate() if roi < threshold else None

    def activate_on_competition(self, features: CompetitorFeatures, limit: float = 0.5) -> Optional[Innovation]:
        prob = self.strategy_bot.predict(features)
        return self.ideate() if prob > limit else None

    def run(self, interval: float = 60.0, iterations: Optional[int] = None) -> None:
        count = 0
        while iterations is None or count < iterations:
            self.workflows = self.workflow_db.load()
            self.ideate()
            count += 1
            time.sleep(interval)

__all__ = ["WorkflowStep", "Workflow", "WorkflowDB", "Innovation", "InnovationsDB", "ContrarianModelBot"]
