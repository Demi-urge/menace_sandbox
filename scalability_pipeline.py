"""Pipeline for assessing scalability and redeploying bots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable

from .bot_development_bot import BotDevelopmentBot, BotSpec
from .bot_testing_bot import BotTestingBot
from .deployment_bot import DeploymentBot, DeploymentSpec
from .scalability_assessment_bot import ScalabilityAssessmentBot, TaskInfo
from vector_service.context_builder import ContextBuilder


@dataclass
class DeploymentResult:
    """Result after scalability checks and deployment."""

    deploy_id: int
    resources: Dict[str, Dict[str, float]]


class ScalabilityPipeline:
    """Iteratively improve bots based on scalability reports."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        developer: BotDevelopmentBot | None = None,
        tester: BotTestingBot | None = None,
        scaler: ScalabilityAssessmentBot | None = None,
        deployer: DeploymentBot | None = None,
        max_iters: int = 3,
    ) -> None:
        if context_builder is None:
            raise ValueError("ContextBuilder is required")
        self.context_builder = context_builder
        self.developer = developer or BotDevelopmentBot(
            context_builder=self.context_builder
        )
        self.tester = tester or BotTestingBot()
        self.scaler = scaler or ScalabilityAssessmentBot()
        self.deployer = deployer or DeploymentBot()
        self.max_iters = max_iters

    # ------------------------------------------------------------------
    def _build_and_test(self, name: str) -> None:
        spec = BotSpec(name=name, purpose=name, functions=["run"])
        self.developer.build_bot(spec, context_builder=self.context_builder)
        self.tester.run_unit_tests([name])

    def _estimate_resources(self, tasks: Iterable[TaskInfo]) -> Dict[str, Dict[str, float]]:
        return {t.name: {"cpu": t.cpu, "memory": t.memory} for t in tasks}

    def run(self, bots: Iterable[str]) -> DeploymentResult:
        names = list(bots)
        blueprint = json.dumps({"tasks": [{"name": n} for n in names]})
        report = self.scaler.analyse(blueprint)

        # iteratively fix scalability issues
        iters = 0
        while report.bottlenecks and iters < self.max_iters:
            iters += 1
            for name in report.bottlenecks:
                self._build_and_test(name)
            report = self.scaler.analyse(blueprint)

        resources = self._estimate_resources(report.tasks)
        spec = DeploymentSpec(name="scalable", resources=resources, env={})

        dep_id = self.deployer.deploy("scalable", names, spec)
        rec = self.deployer.db.get(dep_id)
        attempts = 0
        while rec.get("status") != "success" and attempts < self.max_iters:
            attempts += 1
            self.developer.errors.append("deployment failed")
            for name in names:
                self._build_and_test(name)
            dep_id = self.deployer.deploy("scalable", names, spec)
            rec = self.deployer.db.get(dep_id)

        return DeploymentResult(deploy_id=dep_id, resources=resources)


__all__ = ["DeploymentResult", "ScalabilityPipeline"]
